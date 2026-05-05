# NVBug 5615248 — Reproducer

Confirms that the piecewise-CUDA-graph regression described in comments
\#15 and \#18 of NVBug 5615248 still exists on `main`.

## Bug summary

With `enable_piecewise_cuda_graph: True` and `max_seq_len = 128`,
the largest CUDA graph that gets *recorded* during piecewise warmup is
`num_tokens = 64` (it should reach `num_tokens = 128`). Any prefill chunk
with `ISL > 64` falls back to eager execution. Workaround: set
`max_seq_len = 129` (next-power-of-two + 1).

Root cause is two off-by-ones in `tensorrt_llm/_torch/pyexecutor/model_engine.py`
interacting with the buggy power-of-2 default in
`TorchCompileConfig.set_default_capture_num_tokens`
(`tensorrt_llm/llmapi/llm_args.py:3523-3528`). See chat thread for full trace.

## Prerequisites

- A GPU (any modern arch — bug is GPU-architecture-independent, customer used L40S).
- A working TRT-LLM Python environment built from `main`.
  Recommended: a TRT-LLM dev container, or `pip install build/tensorrt_llm-*.whl`
  on a GPU node.
- TinyLlama weights. Adjust the `MODEL` path in the commands below if
  yours is elsewhere. Default assumed:

  ```
  /home/bbuddharaju/scratch/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0
  ```

## Steps

### Step 1 — Confirm the bug (max_seq_len = 128)

```bash
cd /home/bbuddharaju/scratch/TensorRT-LLM/nvbugs_5615248

MODEL=/home/bbuddharaju/scratch/llm-models/llama-models-v2/TinyLlama-1.1B-Chat-v1.0

trtllm-serve "$MODEL" \
    --backend pytorch \
    --port 8000 \
    --extra_llm_api_options extra_llm_options_buggy.yaml \
    2>&1 | tee serve_buggy.log
```

Wait until the warmup phase prints `Running piecewise CUDA graph warmup...`
followed by several `Run piecewise CUDA graph warmup for num tokens=...` lines.
Once you see the server start listening (or the warmup section finishes), you
can `Ctrl+C` it.

In another shell (or after stopping the server):

```bash
grep -E "Running piecewise CUDA graph warmup|Run piecewise CUDA graph warmup for num tokens" serve_buggy.log
```

**Expected (buggy main):**

```
Running piecewise CUDA graph warmup...
Run piecewise CUDA graph warmup for num tokens=64
Run piecewise CUDA graph warmup for num tokens=32
Run piecewise CUDA graph warmup for num tokens=16
Run piecewise CUDA graph warmup for num tokens=8
Run piecewise CUDA graph warmup for num tokens=4
Run piecewise CUDA graph warmup for num tokens=2
Run piecewise CUDA graph warmup for num tokens=1
```

Note the absence of `num tokens=128`. That is the bug:
- `_piecewise_cuda_graph_num_tokens` *includes* 128 (because `128 <= max_num_tokens=128*1=128`).
- But `_create_warmup_request(num_tokens=128, num_gen_tokens=0)` returns `None`
  inside `_capture_piecewise_cuda_graphs` because `max_context_requests * (max_seq_len - 1) = 1 * 127 < 128`.
- The warmup loop silently `continue`s, so no CUDA graph is recorded for 128.
- At runtime, `_get_padding_params` happily pads prefill chunks with `<= 128` tokens
  to 128, but the inner `PiecewiseRunner` finds `entry.cuda_graph is None` for the
  128 entry and falls into `default_callable(*args)` (eager). Net effect:
  any prefill with `64 < ISL <= 128` runs eager instead of piecewise.

### Step 2 — Confirm the workaround (max_seq_len = 129)

```bash
trtllm-serve "$MODEL" \
    --backend pytorch \
    --port 8000 \
    --extra_llm_api_options extra_llm_options_workaround.yaml \
    2>&1 | tee serve_workaround.log
```

After warmup:

```bash
grep -E "Running piecewise CUDA graph warmup|Run piecewise CUDA graph warmup for num tokens" serve_workaround.log
```

**Expected (workaround):**

```
Running piecewise CUDA graph warmup...
Run piecewise CUDA graph warmup for num tokens=128
Run piecewise CUDA graph warmup for num tokens=64
Run piecewise CUDA graph warmup for num tokens=32
Run piecewise CUDA graph warmup for num tokens=16
Run piecewise CUDA graph warmup for num tokens=8
Run piecewise CUDA graph warmup for num tokens=4
Run piecewise CUDA graph warmup for num tokens=2
Run piecewise CUDA graph warmup for num tokens=1
```

The `num tokens=128` line is the proof: with `max_seq_len=129`,
`_create_warmup_request(num_tokens=128)` succeeds because
`max_seq_len - 1 = 128 >= 128`.

### Step 3 (optional) — Observe the eager fall-back at request time

To confirm a real ISL=107 request goes through the eager `default_callable`
in the buggy case, restart the buggy server with debug logging enabled and send
a single 107-token prompt. The `Pad tensor with 107 tokens to 128 tokens` debug
line will print, but no CUDA-graph replay will follow for the 128 piece:

```bash
TLLM_LOG_LEVEL_BY_MODULE="debug:_torch.compilation,_torch.pyexecutor;info:" \
trtllm-serve "$MODEL" \
    --backend pytorch \
    --port 8000 \
    --extra_llm_api_options extra_llm_options_buggy.yaml \
    2>&1 | tee serve_buggy_debug.log &

SERVER_PID=$!
sleep 60   # adjust for warmup time

# Send a 107-token-ish prompt
curl -s http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
          "model": "TinyLlama",
          "prompt": "'"$(python3 -c 'print(("hello " * 100).strip())')"'",
          "max_tokens": 20,
          "temperature": 0.0
        }' >/dev/null

kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

grep -E "Pad tensor with .* tokens to .* tokens|Piecewise CUDA graph cannot be used" serve_buggy_debug.log
```

You should see lines like `Pad tensor with 107 tokens to 128 tokens` confirming
the outer padding fired, while the actual CUDA graph for 128 was never captured
(only the 1..64 ones were).

## Pass/fail criteria

| Check | Buggy expected | Workaround expected |
|---|---|---|
| Largest `Run piecewise CUDA graph warmup for num tokens=N` | N = 64 | N = 128 |
| `num tokens=128` line present | NO | YES |

If you see the buggy column on `main`, the regression is confirmed.
