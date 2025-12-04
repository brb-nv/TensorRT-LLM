# Helix Parallelism

For more details, see the following paper describing Helix Parallelism:
[Helix Parallelism](https://arxiv.org/pdf/2507.07120).

Helix parallelism is a type of context / KV cache parallelism.
Unlike most other types of context parallelism (e.g. star attention or ring attention),
Helix is used during the decode/generation phase only.

Helix parallelism is most useful in scenarios where all of the following
conditions apply:

1. Disaggregated serving: Helix parallelism will be applied to the generation
   server(s) only.
2. High input sequence length / context size: depending on the model, Helix
   likely only provides performance advantages with input sequence lengths >64K,
   possibly more.
3. Low batch sizes: Helix is most useful in low-latency / high tokens/s/user
   scenarios. On the typical Pareto curve, these are found at the highest point
   on the x-axis / towards the right of the plot.

## Testing Helix with TensorRT-LLM

There are currently two main ways of testing Helix parallelism in TensorRT-LLM, as described below.

### Correctness test for the MLA module

The simplest test can be found in
[test_mla_helix.py](../../../../tests/unittest/_torch/modules/test_mla_helix.py).

This is a unit test of just the
[MLA attention module](../../../../tensorrt_llm/_torch/modules/attention.py),
which has been updated to support Helix parallelism.

### Accuracy benchmarking with Deepseek V3 Lite

This e2e accuracy test evaluates Deepseek V3 Lite in disaggregated mode on MMLU & GSM8K benchmarks with helix parallelism.

The test can be found in [test_disaggregated_serving.py](../../../../tests/integration/defs/accuracy/test_disaggregated_serving.py) under the name `TestDeepSeekV3Lite::test_auto_dtype_with_helix`.

This shows an example of how to configure generation servers with helix parallelism. Specifically, the following fields:

```json
"context_parallel_size": 2,
"cp_config": {
      "cp_type": "HELIX",
      "tokens_per_block": 32
},
```

`cp_config.tokens_per_block` config must match `kv_cache_config.tokens_per_block` in LLMAPI.
