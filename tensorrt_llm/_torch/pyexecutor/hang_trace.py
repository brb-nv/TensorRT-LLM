# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-process breadcrumb ring for diagnosing multi-rank hangs.

Motivation: when a tensor-parallel rank group wedges, every rank ends up
blocked in ``cuLaunchKernel`` once its launch queue backs up, so neither the
Python stack nor ``cuda-gdb`` says which rank stopped participating first --
cuda-gdb cannot even stop a thread spinning in the driver. What distinguishes
a collective mismatch from a merely slow rank is the *sequence of collectives
each rank issued*: ranks that agree on the sequence are skewed, ranks that
disagree are deadlocked, and the first counter that differs names the
offending call.

So this keeps two things per rank: a monotonic counter per collective kind,
and a bounded ring of recent breadcrumbs (layer, phase, batch shape). Diffing
the counters and the ring tail across the ranks of a wedged node is the
intended workflow.

Everything is off unless ``TLLM_HANG_TRACE=1``. When off, ``enabled()`` is a
module-global bool read and every entry point returns immediately, so probes
can sit on the hot path.

Deliberate constraints on probes, because these run inside the model forward:
  - Never synchronize. No ``.item()``, ``.cpu()``, or ``torch.cuda.synchronize``
    in a probe: that would perturb the very timing being measured and can mask
    a launch-queue stall entirely.
  - Never format at record time. Arguments are stashed raw and rendered only
    in :func:`dump`, which runs once, after the hang.

Environment:
    ``TLLM_HANG_TRACE=1``            enable recording.
    ``TLLM_HANG_TRACE_CAPACITY``     ring size in entries (default 65536).
    ``TLLM_HANG_TRACE_DIR``          write dumps here, one file per rank, so
                                     a wedged node can be harvested even if
                                     the log is truncated by the hard kill.
    ``TLLM_HANG_TRACE_DUMP_EVERY_S`` also dump on this interval, so the trace
                                     survives a kill that outruns the detector.
"""

import os
import signal
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

_ENABLED = os.environ.get("TLLM_HANG_TRACE") == "1"

# Sized for the fully-instrumented model: ~12 breadcrumbs per layer per
# forward across ~60 layers is ~700 entries an iteration, so this holds
# roughly 90 iterations -- far more run-up than the detector window needs,
# for a few MB of RSS.
try:
    _CAPACITY = max(64, int(os.environ.get("TLLM_HANG_TRACE_CAPACITY", "65536")))
except ValueError:
    _CAPACITY = 65536

# Preallocated so steady-state recording never grows a list or allocates a
# slot object; only the per-event tuple is created.
_ring: list = [None] * _CAPACITY
_next: int = 0

# tag -> count. The divergence detector: compare these across ranks.
_counters: dict = {}

# Ambient context, set by the executor and the model so probes stay cheap and
# callers do not have to thread rank/iteration/phase through every signature.
_rank: int = -1
_iteration: int = -1
_phase: str = "init"

# Guards only dump(), which walks the ring from the detector thread (or a
# signal handler) while the executor thread may still be appending. Recording
# stays lock-free: appends are single-writer in practice (the executor loop),
# and the GIL makes the individual slot store and index bump atomic enough
# that a rare interleaving costs at most one misordered breadcrumb -- an
# acceptable trade for keeping the probe off the lock path.
_dump_lock = threading.Lock()


# Resolved once, at import: an ``import`` statement inside a probe is one more
# thing for Dynamo to choke on, and this module is imported from code that has
# already imported torch. Falling back to a constant False keeps the module
# importable (and unit testable) without torch.
try:
    import torch as _torch

    _is_compiling = _torch.compiler.is_compiling
except Exception:  # noqa: BLE001 - torch is optional for tests
    def _is_compiling() -> bool:
        return False


def _tracing() -> bool:
    """True while Dynamo is tracing this code."""
    try:
        return _is_compiling()
    except Exception:  # noqa: BLE001 - a probe must never raise
        return False


def enabled() -> bool:
    """True when ``TLLM_HANG_TRACE=1``, regardless of compilation state."""
    return _ENABLED


def active() -> bool:
    """True when probes should run: enabled *and* not being traced.

    Call sites must gate on this rather than :func:`enabled`, because the
    context role runs under ``torch.compile`` and Dynamo traces straight into
    the probe. Two things go wrong if it does. It hits ``time.monotonic_ns()``
    and dies with "Attempted to call function marked as skipped" (this took
    out every context worker in jobs 2993589-2993592). And even if it could
    trace it, evaluating probe arguments such as ``hidden_states.shape[0]``
    during tracing introduces symbolic guards that break piecewise CUDA graph
    capture -- the same hazard ``gated_mlp.py`` guards against.

    Gating on a function that Dynamo folds to a constant makes it prune the
    whole probe, arguments included, so compiled graphs come out byte-identical
    to an uninstrumented build. The probes still fire in eager execution, which
    is where every captured wedge has actually been sitting.
    """
    return _ENABLED and not _tracing()


def set_rank(rank: int) -> None:
    """Tag subsequent breadcrumbs with this rank."""
    global _rank
    _rank = rank


def set_iteration(iteration: int) -> None:
    """Tag subsequent breadcrumbs with the executor iteration in flight."""
    global _iteration
    _iteration = iteration


def set_phase(phase: str) -> str:
    """Tag subsequent breadcrumbs with a coarse phase; returns the previous one.

    Phase is what separates the target forward from each speculative draft
    step. Since disabling speculation is the one change that stops the hang,
    every breadcrumb needs to say which side of that boundary it came from.
    Callers restore the previous value so nesting works.
    """
    global _phase
    previous = _phase
    _phase = phase
    return previous


@contextmanager
def phase(name: str):
    """Scope breadcrumbs to ``name``, restoring the previous phase on exit.

    Entered once per model forward or draft step, never per layer, so the
    context-manager overhead does not matter here.
    """
    if not _ENABLED or _tracing():
        yield
        return
    previous = set_phase(name)
    record("phase_enter", name)
    try:
        yield
    finally:
        record("phase_exit", name)
        set_phase(previous)


def record(tag: str, a: Any = None, b: Any = None, c: Any = None, d: Any = None) -> None:
    """Append one breadcrumb. Values are stored raw and rendered by :func:`dump`.

    The tracing guard must precede ``time.monotonic_ns()``: Dynamo cannot call
    it and raises "Attempted to call function marked as skipped".
    """
    if not _ENABLED or _tracing():
        return
    global _next
    idx = _next
    _next = idx + 1
    _ring[idx % _CAPACITY] = (time.monotonic_ns(), _iteration, _phase, tag, a, b, c, d)


def bump(tag: str) -> int:
    """Increment and return this rank's counter for ``tag``.

    The returned value is the sequence number of *this* occurrence, so it can
    be recorded alongside the breadcrumb and lined up against peer ranks.
    """
    if not _ENABLED or _tracing():
        return -1
    count = _counters.get(tag, 0) + 1
    _counters[tag] = count
    return count


def record_collective(tag: str, numel: Any = None, extra: Any = None) -> None:
    """Record a collective and its per-rank sequence number.

    Every collective goes through here so the counters stay directly
    comparable across ranks; ``tag`` distinguishes the kind (allreduce,
    allgather, ...) because a mismatch in *which* collective was issued is as
    informative as a mismatch in how many.
    """
    if not _ENABLED or _tracing():
        return
    record(tag, bump(tag), numel, extra)


def counters() -> dict:
    """Snapshot of every counter, for the hang dump."""
    return dict(_counters)


def dump(limit: int = 4096) -> str:
    """Render the counters and the newest ``limit`` breadcrumbs, oldest first."""
    if not _ENABLED:
        return "hang_trace: disabled (set TLLM_HANG_TRACE=1 to enable)"
    with _dump_lock:
        total = _next
        snapshot = list(_ring)
        counts = dict(_counters)
        rank = _rank
        iteration = _iteration
        phase = _phase

    lines = [
        f"HANG_TRACE rank={rank} pid={os.getpid()} iteration={iteration} "
        f"phase={phase} recorded={total} capacity={_CAPACITY}"
    ]
    # Printed on one line per rank on purpose: this is the row an operator
    # diffs across the four ranks of a wedged node.
    lines.append(f"HANG_TRACE counters rank={rank}: " + (
        " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"))

    count = min(limit, total, _CAPACITY)
    if count == 0:
        return "\n".join(lines)

    entries = [snapshot[i % _CAPACITY] for i in range(total - count, total)]
    entries = [e for e in entries if e is not None]
    if not entries:
        return "\n".join(lines)
    # Relative microseconds: the question is how long the rank sat at its last
    # breadcrumb, not when the run started.
    newest_ns = entries[-1][0]
    for t_ns, it, ph, tag, a, b, c, d in entries:
        args = " ".join(str(v) for v in (a, b, c, d) if v is not None)
        lines.append(
            f"HANG_TRACE rank={rank} t-{(newest_ns - t_ns) / 1000.0:12.1f}us "
            f"iter={it} {ph} {tag} {args}".rstrip())
    return "\n".join(lines)


def dump_to_file(reason: str = "manual") -> Optional[str]:
    """Write the dump under ``TLLM_HANG_TRACE_DIR``; return the path or None.

    Worth having in addition to logging it: the hard kill and the launcher's
    teardown can truncate stdout, and a per-rank file is also far easier to
    diff across the ranks of a node than four interleaved log streams.
    """
    if not _ENABLED:
        return None
    directory = os.environ.get("TLLM_HANG_TRACE_DIR")
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        # Hostname is part of the name because each context server is its own
        # MPI job: rank 0 exists on every node, so rank alone would collide
        # when several replicas share a dump directory.
        path = os.path.join(
            directory,
            f"hangtrace_{socket.gethostname()}_rank{_rank}_pid{os.getpid()}.txt")
        # Truncate rather than append: only the newest snapshot is wanted, and
        # a periodic dumper would otherwise grow without bound.
        with open(path, "w") as handle:
            handle.write(f"# reason={reason} wall={time.time()}\n")
            handle.write(dump())
            handle.write("\n")
        return path
    except OSError:
        return None


def _handle_dump_signal(signum, frame) -> None:  # noqa: ARG001 - signal API
    """SIGUSR1 -> dump. Lets a watcher harvest a rank the detector has not killed."""
    try:
        from tensorrt_llm.logger import logger

        logger.error(dump())
    except Exception:  # noqa: BLE001 - a diagnostic handler must never raise
        pass
    dump_to_file(reason="sigusr1")


def _periodic_dump_loop(interval: float) -> None:
    while True:
        time.sleep(interval)
        dump_to_file(reason="periodic")


def install() -> None:
    """Install the SIGUSR1 dump hook and the optional periodic dumper.

    Called once from the executor. Signal registration only works on the main
    thread, so failures are tolerated: the detector-driven dump is the primary
    path and this is a convenience on top of it.
    """
    if not _ENABLED:
        return
    try:
        signal.signal(signal.SIGUSR1, _handle_dump_signal)
    except (ValueError, OSError):
        pass  # not the main thread, or signals unavailable
    try:
        interval = float(os.environ.get("TLLM_HANG_TRACE_DUMP_EVERY_S", "0"))
    except ValueError:
        interval = 0.0
    if interval > 0:
        threading.Thread(target=_periodic_dump_loop,
                         args=(interval, ),
                         daemon=True,
                         name="hang_trace_dump").start()


def reset() -> None:
    """Clear all state. For tests."""
    global _next, _rank, _iteration, _phase
    with _dump_lock:
        _next = 0
        _rank = -1
        _iteration = -1
        _phase = "init"
        _counters.clear()
        for i in range(_CAPACITY):
            _ring[i] = None


def _set_enabled_for_test(value: bool, capacity: Optional[int] = None) -> None:
    """Toggle the module without re-importing. For tests only."""
    global _ENABLED, _CAPACITY, _ring
    _ENABLED = value
    if capacity is not None:
        _CAPACITY = capacity
        _ring = [None] * capacity
    reset()
