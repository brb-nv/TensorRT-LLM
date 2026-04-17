# Dynamo TRTLLM Recipe Configs

Serving configs extracted from [Dynamo TRTLLM recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes).
Each file is the `ConfigMap` YAML from a Dynamo `deploy.yaml`, usable directly with `trtllm-serve --config`.

## Directory Structure

```
<model>/
├── agg/
│   └── config.yaml                  # Aggregated serving config
└── disagg/
    ├── config-prefill.yaml          # Prefill worker config
    └── config-decode.yaml           # Decode worker config
```

## Recipes

| Model | Mode | GPUs | Dynamo Source |
|---|---|---|---|
| qwen3-32b-fp8 | agg | 2× H100/H200/A100 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml) |
| qwen3-32b-fp8 | disagg | 8× H100/H200/A100 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/disagg/deploy.yaml) |
| qwen3-235b-a22b-fp8 | agg | 16× H100/H200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-235b-a22b-fp8/trtllm/agg/deploy.yaml) |
| qwen3-235b-a22b-fp8 | disagg | 16× H100/H200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-235b-a22b-fp8/trtllm/disagg/deploy.yaml) |
| gpt-oss-120b | agg | 4× GB200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/gpt-oss-120b/trtllm/agg/deploy.yaml) |
| gpt-oss-120b | disagg | 5× Blackwell | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/gpt-oss-120b/trtllm/disagg/deploy.yaml) |
| deepseek-r1 | disagg (WideEP) | 36× GB200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml) |
| nemotron-3-super-fp8 | disagg | 4× H100/H200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3-super-fp8/trtllm/disagg/deploy.yaml) |
| deepseek-v32-fp4 | agg (round-robin) | 32× GB200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v32-fp4/trtllm/agg-round-robin/deploy.yaml) |
| deepseek-v32-fp4 | disagg (kv-router) | 32× GB200 | [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/deepseek-v32-fp4/trtllm/disagg-kv-router/deploy.yaml) |
| kimi-k2.5 | agg | 8× B200 | [baseten](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.5/trtllm/agg/baseten/deploy.yaml) / [nvidia](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.5/trtllm/agg/nvidia/deploy.yaml) |

## Usage

These configs are valid `trtllm-serve --config` YAML files:

```bash
trtllm-serve Qwen/Qwen3-32B-FP8 --config tests/integration/defs/dynamo/configs/qwen3-32b-fp8/agg/config.yaml
```

They can also be used for schema validation testing against `TorchLlmArgs` without any GPU.
