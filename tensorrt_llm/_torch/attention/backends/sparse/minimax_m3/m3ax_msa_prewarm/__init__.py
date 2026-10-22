# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msa-prewarm: compile every fmha_sm100 (MiniMax MSA) variant the serving trace can select during warmup.

Modules (none imports torch or tensorrt_llm at import time):
  hook      entry points DecoderWarmup calls (around its run, and at the start of its MSA indexer prewarm)
  core      variant enumeration (fmha_sm100's own dispatch functions), parallel compile, serve-time observer
  sdwarm    sparse-decode prewarm: the Triton sparse decode is warmed for every local batch and pointer-alignment
            signature an eager step can reach, at the end of warmup (M3AX_SPARSE_DECODE_PREWARM=0: off)
  selftest  coverage check: real fmha_sm100 plans for every q, variant lookups recorded, nothing may be missing

Switches: M3AX_STACK_A=0 or M3AX_MSA_PREWARM_VARIANTS=0 (or false/off/no) turns the prewarm off (both default on).
core also prewarms o12a's balanced split variants when o12a (msa_proxy_balance) is on.
"""

VERSION = "msa-prewarm-1+stack-a"
