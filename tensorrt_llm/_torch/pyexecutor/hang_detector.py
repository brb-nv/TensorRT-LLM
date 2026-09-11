# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Optional

from tensorrt_llm._utils import ENABLE_MULTI_DEVICE, mpi_comm, mpi_disabled, print_all_stacks
from tensorrt_llm.logger import logger

# 137 == 128 + SIGKILL(9): the exit code a shell reports for a SIGKILL'd process.
_HARD_KILL_EXIT_CODE = 137

_DEFAULT_TIMEOUT_SECONDS = 300

# Seconds to stay alive after detection before hard-killing, so an external
# tool (py-spy, cuda-gdb, nvidia-smi) can attach to the still-hung process.
# 0 keeps the original behavior of killing immediately.
_PRE_KILL_DELAY_ENV = "TLLM_HANG_DETECTOR_PRE_KILL_DELAY_S"
_TIMEOUT_ENV = "TLLM_HANG_DETECTION_TIMEOUT_S"

# Opt-in, and deliberately off by default. Sampling the device shells out to
# nvidia-smi, which can add tens of seconds before the hard kill reaches peer
# ranks -- exactly the delay propagate_hard_kill() exists to avoid. Enable it
# only on a job that is being debugged.
_DIAGNOSTICS_ENV = "TLLM_HANG_DETECTOR_DIAGNOSTICS"

# nvidia-smi must never outlive the diagnostic window it is being run inside.
_NVIDIA_SMI_TIMEOUT_SECONDS = 10

# Split into independent queries on purpose: nvidia-smi rejects the whole
# --query-gpu list if any single field is unsupported on the installed driver
# or architecture, and the ECC/row-remap field names differ across
# generations. One failing group must not cost us the others.
_NVIDIA_SMI_QUERIES = (
    (
        "gpu",
        "index,uuid,utilization.gpu,utilization.memory,temperature.gpu,power.draw,"
        "clocks_throttle_reasons.active",
    ),
    ("gpu", "ecc.errors.uncorrected.volatile.total"),
    (
        "gpu",
        "remapped_rows.correctable,remapped_rows.uncorrectable,remapped_rows.pending,"
        "remapped_rows.failure",
    ),
    ("gpu", "retired_pages.double_bit.count,retired_pages.pending"),
    ("compute-apps", "pid,used_memory"),
)


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int from the environment, falling back on any error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not an integer; using {default}.")
        return default
    if value < 0:
        logger.warning(f"{name}={value} is negative; using {default}.")
        return default
    return value


def _best_effort_flush_streams() -> None:
    """Flush stdout/stderr without ever raising; diagnostics must not block hard kill."""
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass


def _best_effort_log_error(message: str) -> None:
    """Log at error level without ever raising; diagnostics must not block hard kill."""
    try:
        logger.error(message)
    except Exception:  # noqa: BLE001 - diagnostics must not block hard kill
        pass


def _log_cuda_stream_state() -> None:
    """Log whether the GPU still owes this process work.

    This is the discriminator a Python stack cannot give: the executor thread
    blocks on ordinary tensor ops once the launch queue backs up, so the frame
    it stops on says nothing about whether the device is stuck. A non-empty
    stream means work was submitted and never retired (device-side stall);
    an empty stream means the device is idle and the hang is host-side.

    ``query()`` is ``cudaStreamQuery``, which does not block, so it is safe to
    call from the detector thread against a wedged device.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            _best_effort_log_error("HangDetector: CUDA unavailable; no device state to report.")
            return
        device = torch.cuda.current_device()
        stream = torch.cuda.current_stream()
        _best_effort_log_error(
            f"HangDetector: device={device} stream={stream} "
            f"stream_idle={stream.query()} "
            f"alloc={torch.cuda.memory_allocated(device)} "
            f"reserved={torch.cuda.memory_reserved(device)}"
        )
    except Exception as e:  # noqa: BLE001 - diagnostics must not block hard kill
        _best_effort_log_error(f"HangDetector: CUDA state probe failed ({e}).")


def _log_nvidia_smi() -> None:
    """Log driver-visible GPU health.

    ``dmesg`` is unreadable from inside the container on many clusters, so Xid
    faults cannot be read directly. These fields are the readable proxy:
    uncorrected ECC counts, retired/pending pages, and active throttle reasons
    distinguish a hardware or driver fault from a software deadlock.
    """
    for name, fields in _NVIDIA_SMI_QUERIES:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-{name}={fields}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            _best_effort_log_error(f"HangDetector: nvidia-smi {fields} failed ({e}).")
            continue
        output = (result.stdout or result.stderr or "").strip()
        _best_effort_log_error(f"HangDetector: nvidia-smi [{fields}]:\n{output}")


def _log_hang_trace() -> None:
    """Log this rank's collective counters and recent breadcrumbs.

    Separately gated from the rest: the trace is pure in-process memory, so
    unlike nvidia-smi it costs nothing to print and is worth having even on a
    run where the device probes are too slow to enable.
    """
    try:
        from .hang_trace import dump, enabled

        if not enabled():
            return
        _best_effort_log_error(dump())
    except Exception as e:  # noqa: BLE001 - diagnostics must not block hard kill
        _best_effort_log_error(f"HangDetector: hang-trace dump failed ({e}).")


def log_hang_diagnostics() -> None:
    """Log the non-blocking device diagnostics available at hang time.

    The breadcrumb trace runs first and on its own gate: it is the signal that
    identifies *which* rank diverged, and it must reach the log even if a
    wedged driver makes the nvidia-smi probes below time out.

    The rest is a no-op unless ``TLLM_HANG_DETECTOR_DIAGNOSTICS=1``; see
    ``_DIAGNOSTICS_ENV`` for why that must not be on by default.
    """
    _log_hang_trace()
    if os.environ.get(_DIAGNOSTICS_ENV) != "1":
        _best_effort_flush_streams()
        return
    _best_effort_log_error(f"HangDetector: host={socket.gethostname()} pid={os.getpid()}")
    _log_cuda_stream_state()
    _log_nvidia_smi()
    _best_effort_flush_streams()


def pre_kill_hold() -> None:
    """Stay alive after detection so external tools can attach.

    The hard kill destroys the only state that names the stuck kernel. When
    ``TLLM_HANG_DETECTOR_PRE_KILL_DELAY_S`` is set, hold the process in its
    hung state for that long first and announce the window on a greppable
    line, so a harness hook can run ``py-spy dump`` or ``cuda-gdb -p`` against
    a live process. Progress is logged each interval so the log shows how much
    of the window remained if the job is torn down from outside.
    """
    delay = _env_int(_PRE_KILL_DELAY_ENV, 0)
    if delay == 0:
        return
    host = socket.gethostname()
    pid = os.getpid()
    _best_effort_log_error(
        f"HANG_DETECTOR_ATTACH_WINDOW_OPEN host={host} pid={pid} seconds={delay} "
        f"-- attach now, e.g. py-spy dump --pid {pid}"
    )
    _best_effort_flush_streams()
    deadline = time.monotonic() + delay
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(30.0, remaining))
        _best_effort_log_error(
            f"HANG_DETECTOR_ATTACH_WINDOW host={host} pid={pid} "
            f"remaining={max(0.0, deadline - time.monotonic()):.0f}s"
        )
        _best_effort_flush_streams()
    _best_effort_log_error(
        f"HANG_DETECTOR_ATTACH_WINDOW_CLOSED host={host} pid={pid} -- proceeding to hard kill"
    )
    _best_effort_flush_streams()


def propagate_hard_kill(exit_code: int = _HARD_KILL_EXIT_CODE) -> None:
    """Hard-kill this rank and propagate the kill to peer ranks.

    Cross-rank propagation is the load-bearing part: a peer blocked in an NCCL
    collective would otherwise hold its GPU until the job's wall-clock pod-kill.

    - Preferred (when safe): ``MPI_Abort`` aborts the whole MPI job in one call.
      Only safe from the detector's daemon thread when MPI was initialized with
      ``MPI_THREAD_MULTIPLE``; guarded by ``Query_thread``.
    - Fallback: self-``SIGKILL``. The launcher (``mpirun`` propagates by default;
      ``srun`` needs ``--kill-on-bad-exit``) then tears down peers.

    All flushing and logging is best-effort: a closed/broken stdout, stderr, or
    logger must never prevent reaching ``MPI_Abort`` or ``os.kill``.
    """
    _best_effort_flush_streams()
    try:
        if ENABLE_MULTI_DEVICE and not mpi_disabled():
            from mpi4py import MPI

            if MPI.Is_initialized() and MPI.Query_thread() == MPI.THREAD_MULTIPLE:
                _best_effort_log_error(
                    "HangDetector: propagating hard-kill to all ranks via MPI_Abort."
                )
                mpi_comm().Abort(exit_code)
                return  # not reached; Abort does not return
    except Exception as e:  # noqa: BLE001 - last-resort path must not raise
        _best_effort_log_error(
            f"HangDetector: MPI_Abort propagation failed ({e}); falling back to self-SIGKILL."
        )
    _best_effort_log_error(
        "HangDetector: self-SIGKILL; relying on the launcher to propagate to peer ranks."
    )
    os.kill(os.getpid(), signal.SIGKILL)


class HangDetector:
    """Watchdog that fires when the executor loop stops checkpointing.

    When ``timeout`` seconds pass without a ``checkpoint()``, all thread stacks
    are dumped, GPU state is sampled, an optional attach window is held open
    (see ``pre_kill_hold``), and then ``on_detected`` runs (the hard-kill +
    cross-rank propagation path).

    Environment:
        ``TLLM_HANG_DETECTION_TIMEOUT_S``: override the default timeout.
        ``TLLM_HANG_DETECTOR_DIAGNOSTICS=1``: sample CUDA stream and
        nvidia-smi state at hang time.
        ``TLLM_HANG_DETECTOR_PRE_KILL_DELAY_S``: seconds to stay hung before
        killing, so external tools can attach.

    The last two are off by default: both delay the cross-rank hard kill,
    which peers blocked in NCCL are waiting on.
    """

    def __init__(
        self, timeout: Optional[int] = None, on_detected: Optional[Callable[[], None]] = None
    ):
        # No caller passes ``timeout`` today (PyExecutor accepts the parameter
        # but _util.py never forwards it), so the environment is the only way
        # to tune the window without a code change.
        self.timeout = (
            timeout
            if timeout is not None
            else _env_int(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_SECONDS) or _DEFAULT_TIMEOUT_SECONDS
        )
        assert self.timeout > 0, "timeout must be greater than 0"
        self.on_detected = on_detected or (lambda: None)
        self.task = None
        self.loop = None
        self.loop_thread = None
        self.lock = threading.Lock()
        self.active = False
        self._detected = False

    def start(self):
        """Enable hang detection."""

        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.active = True
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=run_loop, daemon=True, name="hang_detector_loop")
        self.loop_thread.start()

    async def _detect_hang(self):
        await asyncio.sleep(self.timeout)
        with self.lock:
            self._detected = True
            logger.error(f"Hang detected after {self.timeout} seconds.")
            print_all_stacks()
        # Diagnostics and the attach window run outside the lock: the hold can
        # last minutes, and callers polling detected() must not block on it.
        log_hang_diagnostics()
        pre_kill_hold()
        self.on_detected()

    def detected(self):
        """Return True if hang is detected."""
        with self.lock:
            return self._detected

    def checkpoint(self):
        """Reset hang detection timer."""
        self.cancel_task()
        if self.active:
            self.task = asyncio.run_coroutine_threadsafe(self._detect_hang(), self.loop)

    def cancel_task(self):
        """Cancel the hang detection task."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
            self.task = None

    @contextmanager
    def pause(self):
        """Pause hang detection in scope."""
        try:
            self.cancel_task()
            yield
        finally:
            self.checkpoint()

    def stop(self):
        """Stop hang detection."""
        self.active = False
        self.cancel_task()
        if self.loop is not None:
            # Cancel all pending tasks before stopping the loop
            def cancel_all_tasks():
                for task in asyncio.all_tasks(self.loop):
                    if not task.done():
                        task.cancel()
                self.loop.call_soon(self.loop.stop)

            self.loop.call_soon_threadsafe(cancel_all_tasks)

            if self.loop_thread is not None and self.loop_thread.is_alive():
                self.loop_thread.join()

            self.loop = None
            self.loop_thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False
