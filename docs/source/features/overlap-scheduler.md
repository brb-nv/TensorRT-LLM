# Overlap Scheduler

To maximize GPU utilization, the scheduler overlaps CPU tasks (e.g., checking sampling stop criteria, updating responses, scheduling the next batch) with GPU computation.

## How It Works

At step *n*, the system launches GPU computation for step *n+1* without waiting for CPU tasks (e.g., stop criteria checks) from step *n* to complete. This allows:

- CPU work (step *n*) and GPU computation (step *n+1*) to run concurrently.
- Better GPU occupancy by reducing idle time.

This concurrent execution pipeline is illustrated in the `PyExecutor`'s logic:

```python
# Schedule and launch GPU forward work for the current step (n).
scheduled_batch, _, _ = self._schedule()
batch_outputs = self._forward_step(scheduled_batch, previous_tensors_device)

# Materialize step (n-1)'s sampled tokens, then emit step (n-1)
# responses to clients before launching step (n) sampling so responses
# reach clients at the earliest possible time. Termination and
# KV-cache bookkeeping are deferred.
if self.previous_batch is not None:
    self._update_requests(self.previous_batch.sample_state)
    pending = self._emit_previous_batch_responses()

# Launch GPU sampling work for the current step (n).
sample_state = self._sample_async(scheduled_batch, batch_outputs)

# Finalize step (n-1): terminate finished requests, update KV-cache
# resources, record iter stats.
if self.previous_batch is not None:
    self._finalize_previous_batch(*pending)
```

## Tradeoff

The optimization introduces one extra decoding step but significantly improves throughput.

## Usage

Enabled by default. To disable, set `disable_overlap_scheduler=True` in the configuration.


## References

- [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)
- https://lmsys.org/blog/2024-12-04-sglang-v0-4/#zero-overhead-batch-scheduler
