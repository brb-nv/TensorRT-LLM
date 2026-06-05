# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-Python utilities for computing per-request performance metrics.

Kept in tensorrt_llm/metrics/ so that this logic has no GPU / heavy
dependencies and can be unit-tested without a full TensorRT-LLM install.
"""

from typing import Optional

from .enums import MetricNames, RequestEventTiming


def process_req_perf_metrics(
    req_perf_metrics_dict: Optional[dict], output_length: int
) -> dict[MetricNames, float | int]:
    """Compute derived per-request latency and token-count metrics.

    Args:
        req_perf_metrics_dict: Raw timing dict from the executor, keyed by
            ``RequestEventTiming`` enum members.  May be ``None`` or empty.
        output_length: Number of output tokens generated for this request.

    Returns:
        Dict mapping ``MetricNames`` enum members to numeric values.
        Keys with value <= 0 are filtered out, except ``REQUEST_QUEUE_TIME``
        which may be 0 (meaning the request was scheduled immediately).
    """
    if not req_perf_metrics_dict:
        return {}

    arrival = req_perf_metrics_dict.get(RequestEventTiming.ARRIVAL_TIME, 0)
    first_scheduled = req_perf_metrics_dict.get(RequestEventTiming.FIRST_SCHEDULED_TIME, 0)
    first_token = req_perf_metrics_dict.get(RequestEventTiming.FIRST_TOKEN_TIME, 0)
    last_token = req_perf_metrics_dict.get(RequestEventTiming.LAST_TOKEN_TIME, 0)

    stat: dict[MetricNames, float | int] = {}

    # Base latency metrics — only compute when all required timestamps are
    # present (> 0).  Absent timestamps default to 0, so a difference that
    # would be negative or zero indicates the timestamp was missing.
    if first_token > 0 and arrival > 0:
        stat[MetricNames.TTFT] = first_token - arrival
    if last_token > 0 and arrival > 0:
        stat[MetricNames.E2E] = last_token - arrival
    # REQUEST_QUEUE_TIME is >= 0 for normally scheduled requests; zero is a
    # valid value (immediate scheduling) so we include it when both timestamps
    # are present.
    if first_scheduled > 0 and arrival > 0:
        stat[MetricNames.REQUEST_QUEUE_TIME] = first_scheduled - arrival

    # Phase latency metrics — require all three anchor timestamps to be valid.
    # PREFILL_TIME = time from first scheduling to first generated token.
    if first_token > 0 and first_scheduled > 0:
        stat[MetricNames.PREFILL_TIME] = first_token - first_scheduled
    # DECODE_TIME = time from first token to last token (generation phase).
    if last_token > 0 and first_token > 0:
        stat[MetricNames.DECODE_TIME] = last_token - first_token
    # INFERENCE_TIME = first_scheduled → last_token (total execution time).
    if last_token > 0 and first_scheduled > 0:
        stat[MetricNames.INFERENCE_TIME] = last_token - first_scheduled

    # Token counts — recorded per candidate.  When n>1 each candidate has
    # its own timestamps and token stream so metrics are valid per candidate.
    if output_length > 0:
        stat[MetricNames.GENERATION_TOKENS] = output_length

    # TPOT = decode duration per output token.  Requires at least 2 tokens
    # (denominator would be 0 for a single-token output) and both timestamps
    # present (first_token=0 default would produce bogus values).
    if output_length > 1 and first_token > 0 and last_token > 0:
        stat[MetricNames.TPOT] = (last_token - first_token) / (output_length - 1)

    # Filter out non-positive values: negatives indicate clock-skew anomalies
    # and should not be reported; absent timestamps produce 0 which is filtered
    # here except for REQUEST_QUEUE_TIME (which is re-added below if valid).
    result = {k: v for k, v in stat.items() if v > 0}
    # Restore REQUEST_QUEUE_TIME=0 if it was explicitly computed (zero queue
    # time is a valid, meaningful observation).
    if MetricNames.REQUEST_QUEUE_TIME in stat and stat[MetricNames.REQUEST_QUEUE_TIME] >= 0:
        result[MetricNames.REQUEST_QUEUE_TIME] = stat[MetricNames.REQUEST_QUEUE_TIME]
    return result


def format_ttft_breakdown(
    req_perf_metrics_dict: Optional[dict],
    *,
    req_id: object = None,
    role: Optional[str] = None,
) -> Optional[str]:
    """Format a single-line host-side TTFT breakdown for a request.

    Decomposes the host timeline recorded by the engine (only populated when
    ``return_perf_metrics`` is enabled) into the stages that contribute to
    time-to-first-token, so a per-request bottleneck is visible in worker logs:

    - ``queue``: arrival -> first schedule (waiting-queue / capacity gating).
    - ``kv_xfer``: KV-cache transfer span (disagg only; the generation server
      blocks on this before it can produce the first token).
    - ``prefill_compute``: first schedule -> first token. On the generation
      server this span *contains* the KV-transfer wait, so ``kv_xfer`` is
      reported separately to attribute it.
    - ``ttft``: arrival -> first token (end-to-end host TTFT on this worker).

    Args:
        req_perf_metrics_dict: Raw timing dict keyed by ``RequestEventTiming``
            (as produced by the executor). May be ``None`` or empty.
        req_id: Optional request identifier to tag the line with.
        role: Optional disagg role tag (e.g. ``"ctx"`` / ``"gen"``).

    Returns:
        A formatted log line, or ``None`` when there is no usable timing data
        (so the caller can skip emitting an empty line).
    """
    if not req_perf_metrics_dict:
        return None

    arrival = req_perf_metrics_dict.get(RequestEventTiming.ARRIVAL_TIME, 0)
    first_scheduled = req_perf_metrics_dict.get(
        RequestEventTiming.FIRST_SCHEDULED_TIME, 0)
    first_token = req_perf_metrics_dict.get(
        RequestEventTiming.FIRST_TOKEN_TIME, 0)
    kv_start = req_perf_metrics_dict.get(
        RequestEventTiming.KV_CACHE_TRANSFER_START, 0)
    kv_end = req_perf_metrics_dict.get(RequestEventTiming.KV_CACHE_TRANSFER_END,
                                       0)
    kv_size = req_perf_metrics_dict.get(RequestEventTiming.KV_CACHE_SIZE, 0)

    fields: list[str] = []
    if req_id is not None:
        fields.append(f"req_id={req_id}")
    if role is not None:
        fields.append(f"role={role}")

    def _ms(label: str, value: float) -> None:
        fields.append(f"{label}={value * 1000:.2f}ms")

    has_data = False
    if first_token > 0 and arrival > 0:
        _ms("ttft", first_token - arrival)
        has_data = True
    if first_scheduled > 0 and arrival > 0:
        _ms("queue", first_scheduled - arrival)
        has_data = True
    if kv_end > 0 and kv_start > 0:
        _ms("kv_xfer", kv_end - kv_start)
        has_data = True
    if first_token > 0 and first_scheduled > 0:
        _ms("prefill_compute", first_token - first_scheduled)
        has_data = True

    if not has_data:
        return None

    if kv_size > 0:
        fields.append(f"kv_size={kv_size}")
    if arrival > 0:
        fields.append(f"arrival={arrival:.6f}")

    return "[TTFT] " + " ".join(fields)


def request_perf_metrics_to_dict(metrics) -> Optional[dict]:
    """Serialize a ``RequestPerfMetrics`` object to a JSON-friendly dict.

    Mirrors the structure returned by the ``trtllm-serve`` ``/perf_metrics``
    HTTP endpoint (minus the disagg clock-offset correction), so the same
    per-request record can be logged to stdout in deployments that do not
    expose that endpoint (e.g. Dynamo-native, where TRT-LLM's OpenAI server
    is not in-path).

    The object is duck-typed (only attribute access is used) so this stays
    free of binding/GPU imports and is unit-testable with a stub.

    Args:
        metrics: A ``tensorrt_llm.bindings.executor.RequestPerfMetrics``-like
            object, or ``None``.

    Returns:
        A nested dict with ``timing_metrics`` / ``kv_cache_metrics`` (and
        ``speculative_decoding`` when draft tokens were used), or ``None``
        when ``metrics`` is ``None``.
    """
    if metrics is None:
        return None

    timing = metrics.timing_metrics
    kv = metrics.kv_cache_metrics
    spec = metrics.speculative_decoding

    record: dict = {
        "first_iter": metrics.first_iter,
        "last_iter": metrics.last_iter,
        "timing_metrics": {
            "arrival_time": timing.arrival_time.total_seconds(),
            "first_scheduled_time": timing.first_scheduled_time.total_seconds(),
            "first_token_time": timing.first_token_time.total_seconds(),
            "last_token_time": timing.last_token_time.total_seconds(),
        },
        "kv_cache_metrics": {
            "num_total_allocated_blocks": kv.num_total_allocated_blocks,
            "num_new_allocated_blocks": kv.num_new_allocated_blocks,
            "num_reused_blocks": kv.num_reused_blocks,
            "num_missed_blocks": kv.num_missed_blocks,
        },
    }
    if timing.kv_cache_size > 0:
        record["timing_metrics"].update({
            "kv_cache_size":
            timing.kv_cache_size,
            "kv_cache_transfer_start":
            timing.kv_cache_transfer_start.total_seconds(),
            "kv_cache_transfer_end":
            timing.kv_cache_transfer_end.total_seconds(),
        })
    if spec.total_draft_tokens > 0:
        record["speculative_decoding"] = {
            "acceptance_rate": spec.acceptance_rate,
            "total_accepted_draft_tokens": spec.total_accepted_draft_tokens,
            "total_draft_tokens": spec.total_draft_tokens,
        }
    return record
