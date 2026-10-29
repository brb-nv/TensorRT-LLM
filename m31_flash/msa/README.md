<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax MSA patches for MiniMax-M3.1-Flash

TensorRT-LLM vendors the parts of [MiniMax MSA](https://github.com/MiniMax-AI/MSA) (MIT) the MiniMax-M3.1-Flash
recipes use: the MSA-v1 Q8KV4 sparse decode (`TRTLLM_M3_MSA_V1_DECODE=1`) and the Q8KV4 decode indexer for NVFP4
index-K (`sparse_attention_config.indexer_kv_dtype: nvfp4`). They live at
`tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/msa_v1`, managed by `scripts/vendor_sources.py` as the
`msa-v1-q8kv4` entry of `3rdparty/vendor_sources.lock.yaml` (MSA `1e7bf76f`). The wheel ships their sources and the
CUTLASS headers they JIT-compile against, so **no separate MSA install is needed**; TensorRT-LLM no longer imports
the external `inference.msa_v1` package.

The quad4 change below is part of the vendored tree: its hunks are in the entry's persistent patch,
`3rdparty/vendor_patches/msa-v1-q8kv4.patch`, which is what TensorRT-LLM builds from. The standalone patch stays here
for building an external MSA with the same layout (for example for the images that predate the vendoring, or another
engine); TensorRT-LLM does not read it.

| Patch | What it does |
|---|---|
| `0001-q8kv4-decode-quad4-V-scale-layout.patch` | Adds an opt-in "quad4" V block-scale layout to MSA's SM100 Q8KV4 sparse decode: `[token // 4][scale_col][token % 4]` per (page, KV head), the order TensorRT-LLM's NVFP4 cache writer uses. The default `token_major` build is unchanged. |

TensorRT-LLM selects the layout itself: when MSA-v1 decode is on it sets `MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT=quad4`.

## Building an external MSA with the patch (not needed by TensorRT-LLM)

Install MSA at commit `1e7bf76f` on top of the TensorRT-LLM wheel, with the patch applied:

```bash
git clone https://github.com/MiniMax-AI/MSA.git /opt/minimax-msa
cd /opt/minimax-msa
git checkout 1e7bf76f
git submodule update --init --recursive        # CUTLASS at MSA's pinned commit
git apply /path/to/TensorRT-LLM/m31_flash/msa/0001-q8kv4-decode-quad4-V-scale-layout.patch
pip install --no-deps -e /opt/minimax-msa
```

MSA has no ahead-of-time kernels: it JIT-compiles on first use into `$TORCH_EXTENSIONS_DIR`. The quad4 variant is a
separate JIT module (cache tag `_vquad4`), so it does not collide with a cached default build. The vendored copy
behaves the same way.

## Check

```python
from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.msa_v1.attention.decode.q8kv4 import jit
assert "quad4" in jit.V_SCALE_LAYOUTS
```

The patch's diff is identical to the series used for every MSA image measured for these recipes (series sha256
`a20552d3…`, image tag `pa20552d`); only the mail header differs.
