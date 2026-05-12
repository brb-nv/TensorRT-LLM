# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build for the TinyLlama-1.1B prefill megakernel (NVBug 5615248, L40S target).

Modeled on Luce-Org/lucebox-hub `megakernel/setup.py`. Architecture is auto-detected
at build time via `torch.cuda.get_device_capability()` so the same source tree builds
for sm_86 (3090, dev box), sm_89 (L40S, the production target), or sm_90 (H100). The
default fallback if no CUDA device is visible is sm_89.
"""

from __future__ import annotations

import os
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _detect_arch() -> str:
    arch = os.environ.get("MEGAKERNEL_CUDA_ARCH")
    if arch:
        return arch
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_89"


def _int_env(name: str, default: int) -> str:
    return str(int(os.environ.get(name, default)))


arch = _detect_arch()
_sm_match = re.search(r"sm_(\d+)", arch)
target_sm = int(_sm_match.group(1)) if _sm_match else 89

# Defaults: TinyLlama-1.1B prefill at M=128 on L40S.
#   142 SMs on L40S; persistent grid targets 2 blocks/SM.
#   256 threads/block = 8 warps = 4x2 warp grid for a 128x64 block-tile.
#
# Sweep (M6) findings, after fixing the `grid_gemm` 2D-tile bug:
#   * `64_128_32_3_2_4` at 11.10 ms in the raw sweep was BOGUS -- the bug
#     made `grid_gemm` skip every m_tile != 0, so BM=64 silently computed
#     half the GEMM math. Re-validate this config post-fix before using.
#   * `128_64_32_3_4_2` at 12.51 ms is the legitimate winner (1.37x over
#     baseline 17.18 ms). It keeps BM = seq_len (avoids the bug regime
#     entirely) and shrinks BN to 64, which doubles N-tile count for the
#     small-N GEMMs (qkv/o_proj/down_proj).
#
# NUM_BLOCKS = 142 * 2 = 284 matches the `__launch_bounds__(BLOCK_SIZE, 2)`
# we set in `tinyllama_megakernel.cu`. The launcher uses
# `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to cap the grid, so if
# 2 blocks/SM doesn't fit (register pressure), we fall back to 1 block/SM
# at gridDim = num_sms automatically. This is therefore safe to default.
num_blocks = _int_env("MEGAKERNEL_NUM_BLOCKS", 284)
block_size = _int_env("MEGAKERNEL_BLOCK_SIZE", 256)
seq_len = _int_env("MEGAKERNEL_SEQ_LEN", 128)
debug = int(os.environ.get("MEGAKERNEL_DEBUG", 0))

# Tile knobs for the M6 sweep. Defaults are the (post-bug-fix) sweep winner.
tile_bm = _int_env("MEGAKERNEL_BM", 128)
tile_bn = _int_env("MEGAKERNEL_BN", 64)
tile_bk = _int_env("MEGAKERNEL_BK", 32)
tile_ns = _int_env("MEGAKERNEL_NSTAGES", 3)
warp_rows = _int_env("MEGAKERNEL_WARP_ROWS", 4)
warp_cols = _int_env("MEGAKERNEL_WARP_COLS", 2)

# Defines shared between the host (bindings.cpp) and device (.cu) compiles. The
# host file `bindings.cpp` includes `common.cuh` and needs all the TLLAMA_*
# constants too, otherwise it fails with "TLLAMA_HIDDEN not declared".
shared_defs = [
    f"-DNUM_BLOCKS={num_blocks}",
    f"-DBLOCK_SIZE={block_size}",
    f"-DMEGAKERNEL_SEQ_LEN={seq_len}",
    f"-DMEGAKERNEL_BM={tile_bm}",
    f"-DMEGAKERNEL_BN={tile_bn}",
    f"-DMEGAKERNEL_BK={tile_bk}",
    f"-DMEGAKERNEL_NSTAGES={tile_ns}",
    f"-DMEGAKERNEL_WARP_ROWS={warp_rows}",
    f"-DMEGAKERNEL_WARP_COLS={warp_cols}",
    f"-DTARGET_SM={target_sm}",
    "-DTLLAMA_HIDDEN=2048",
    "-DTLLAMA_INTERMEDIATE=5632",
    "-DTLLAMA_NUM_HEADS=32",
    "-DTLLAMA_NUM_KV_HEADS=4",
    "-DTLLAMA_HEAD_DIM=64",
    "-DTLLAMA_NUM_LAYERS=22",
    "-DTLLAMA_VOCAB=32000",
    "-DTLLAMA_ROPE_THETA=10000.0f",
    "-DTLLAMA_RMS_EPS=1e-5f",
]

nvcc_args = [
    "-O3",
    f"-arch={arch}",
    # NOTE: deliberately NOT --use_fast_math. With fast-math enabled, nvcc
    # substitutes __sincosf / __expf / approximate rsqrtf for the precise
    # variants, which poisons RoPE (large angles at pair_idx=0 lose precision
    # in range reduction) and RMSNorm's rsqrtf. Empirically that pushed the
    # 22-layer BF16 max_abs vs PyT reference from ~0.04 (expected noise floor)
    # to ~0.15. We may revisit per-stage if perf demands it (M6).
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    *shared_defs,
]
if debug:
    nvcc_args += ["-G", "-DMEGAKERNEL_DEBUG=1"]

cxx_args = [
    "-O3",
    "-std=c++17",
    *shared_defs,
]

sources = [
    "csrc/bindings.cpp",
    "csrc/tinyllama_megakernel.cu",
    "csrc/gemm_pipeline.cu",
    "csrc/stage_tests.cu",
]

setup(
    name="lucebox_tinyllama",
    version="0.1.0",
    description="Persistent single-dispatch prefill megakernel for TinyLlama-1.1B on L40S",
    packages=["lucebox_tinyllama"],
    package_dir={"lucebox_tinyllama": "python/lucebox_tinyllama"},
    ext_modules=[
        CUDAExtension(
            name="lucebox_tinyllama._C",
            sources=sources,
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
