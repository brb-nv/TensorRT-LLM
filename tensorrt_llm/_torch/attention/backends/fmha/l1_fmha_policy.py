# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""l1-fmha (M3AX GB300 stack): a per-graph-batch "virtual SM count" for the trtllm-gen decode FMHA.

MiniMax-M3's full-attention decode (target layers 0-2 through trtllm_gen_dense_decode, and every Eagle3 draft
pass through FlashInferTrtllmGenFmha.run_generation) calls flashinfer.decode.trtllm_batch_decode_with_kv_cache
(backend "trtllm-gen"). That wrapper reads ``sm_count = get_device_sm_count(query.device)`` and hands it to the
FlashInfer launcher as ``mMultiProcessorCount``, which sets the multi-CTA KV split
``numCtasPerSeqKv = min(maxNumCtasPerSeqKv, max(1, MP / (numCtasX * numCtasY * numCtasZ)))``
(fmhaKernels.cuh:648-651) and sizes the partial buffers (kernelParams.h:859-863) and the counter check
(trtllm_fmha_kernel_launcher.cu:266-276) consistently. With one long request per step the launch time is the
longest request's KV divided by the CTAs it gets, so a larger split (more waves) cuts it.

This module holds a measured table {site: {q_len: {graph batch: sm_eff}}} and runs one call with
flashinfer.decode.get_device_sm_count returning ``sm_eff`` for exactly that call (thread-local; the replaced symbol
returns the original function's value everywhere else). Nothing is replaced while the switch is off or the table
is empty, so those processes run the default code path exactly.

Switches (env, read once at import):
  M3AX_STACK_A        GB300 stack master switch, default on; 0/false/off/no = as M3_L1_FMHA=0
  M3_L1_FMHA          default on; 0/false/off/no = off (bitwise stock: the FlashInfer symbol is never replaced)
  M3_L1_FMHA_SITES    comma list of call sites to apply the table to (default "dense,draft")
  M3_L1_FMHA_TABLE    path of a table JSON (bench / dev only) that replaces the built-in table
  M3_L1_FMHA_PREWARM  default on; 0 = no startup prewarm of the table's kernels

Safety: an entry is used only for the geometry it was measured on (heads, KV heads, head dim, FP8 Q/KV, page size,
physical SM count) and only when the call's workspace and counter buffers hold what that sm_eff needs (the launcher
carves its scratch from the workspace without a bounds check, launcher.cu:300-304); otherwise the call runs stock.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading

_FALSE = {"0", "false", "off", "no"}


def _env_on(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in _FALSE


ENABLED = _env_on("M3AX_STACK_A") and _env_on("M3_L1_FMHA")
SITES = frozenset(s.strip() for s in os.environ.get("M3_L1_FMHA_SITES", "dense,draft").split(",") if s.strip())
TABLE_ENV = os.environ.get("M3_L1_FMHA_TABLE", "").strip()
PREWARM = _env_on("M3_L1_FMHA_PREWARM")

# max_seq_len ladder of the startup prewarm (the kernel choice depends on it through maxNumCtasPerSeqKv and the
# tileSizeQ cost model; CUDA graphs bake the capture-time bound, eager mixed steps pass the step's own maximum).
PREWARM_MAX_SEQ_LENS = (128, 512, 2048, 8192, 32768, 131072, 1048576)
# Bytes per partial row the scratch must hold: float2 (max, sum) stats + headDimV partial O at 4 bytes per element
# (an upper bound; the partial O dtype is the cubin's choice, kernelParams.h:98 is a void*).
_STATS_BYTES = 8
_PARTIAL_O_ELT = 4
_SLACK = 4096

# BEGIN GENERATED TABLE (tools/write_table.py; do not edit by hand)
TABLE_SOURCE = 'dev job 3416372 (2026-10-21 11:20 ET, theia0174): test/runs/dev_3416372/bench/table.json = derive_table.py --rule agentx over the GRID=fine bench'
TABLE_GEOMETRY = {'num_heads': 32, 'num_kv_heads': 2, 'head_dim': 128, 'page_size': 128, 'sm_count': 152}
DEFAULT_TABLE = {'dense': {4: {1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 0,
               7: 0,
               8: 0,
               9: 288,
               10: 300,
               11: 286,
               12: 288,
               13: 286,
               14: 448,
               15: 300,
               16: 448,
               17: 272,
               18: 288,
               19: 304,
               20: 440,
               21: 420,
               22: 440,
               23: 414,
               24: 432,
               25: 400,
               26: 572,
               27: 540,
               29: 580,
               30: 600,
               32: 448,
               35: 560,
               40: 880,
               45: 900},
           6: {1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 0,
               7: 0,
               8: 0,
               9: 0,
               10: 0,
               11: 0,
               12: 288,
               13: 286,
               14: 280,
               15: 300,
               17: 272,
               19: 304,
               21: 294,
               23: 276,
               25: 300}},
 'draft': {1: {1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 288,
               7: 378,
               8: 0,
               9: 432,
               10: 440,
               11: 396,
               12: 432,
               13: 442,
               14: 448,
               15: 450,
               16: 448,
               17: 442,
               18: 504,
               19: 456,
               20: 480,
               21: 504,
               22: 528,
               23: 552,
               24: 576,
               25: 600,
               26: 624,
               27: 594,
               29: 580,
               30: 600,
               32: 640,
               35: 560,
               40: 880,
               45: 900},
           4: {1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 0,
               7: 0,
               8: 0,
               9: 288,
               10: 300,
               11: 286,
               12: 288,
               13: 286,
               14: 448,
               15: 300,
               16: 448,
               17: 272,
               18: 288,
               19: 304,
               20: 440,
               21: 420,
               22: 440,
               23: 414,
               24: 432,
               25: 400,
               26: 572,
               27: 540,
               29: 580,
               30: 600,
               32: 448,
               35: 560,
               40: 880,
               45: 900},
           6: {1: 0,
               2: 0,
               3: 0,
               4: 0,
               5: 0,
               6: 0,
               7: 0,
               8: 0,
               9: 0,
               10: 0,
               11: 0,
               12: 288,
               13: 286,
               14: 280,
               15: 300,
               17: 272,
               19: 304,
               21: 294,
               23: 276,
               25: 300}}}
DEFAULT_META = {'1:10': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7438,
          'save_us_S1m': 79.0,
          'save_us_S2': 13.1,
          'save_us_primary': 40.4,
          'sm': 440,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:11': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7218,
          'save_us_S1m': 104.8,
          'save_us_S2': 19.7,
          'save_us_primary': 49.1,
          'sm': 396,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:12': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7592,
          'save_us_S1m': 96.8,
          'save_us_S2': 11.7,
          'save_us_primary': 45.8,
          'sm': 432,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:13': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7232,
          'save_us_S1m': 126.9,
          'save_us_S2': 19.3,
          'save_us_primary': 60.2,
          'sm': 442,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:14': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7092,
          'save_us_S1m': 133.4,
          'save_us_S2': 23.2,
          'save_us_primary': 64.6,
          'sm': 448,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:15': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7233,
          'save_us_S1m': 131.4,
          'save_us_S2': 18.9,
          'save_us_primary': 62.9,
          'sm': 450,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:16': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6912,
          'save_us_S1m': 161.1,
          'save_us_S2': 28.2,
          'save_us_primary': 81.9,
          'sm': 448,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:17': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7143,
          'save_us_S1m': 164.0,
          'save_us_S2': 24.6,
          'save_us_primary': 78.0,
          'sm': 442,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:18': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7016,
          'save_us_S1m': 172.0,
          'save_us_S2': 21.0,
          'save_us_primary': 81.8,
          'sm': 504,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:19': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7217,
          'save_us_S1m': 166.2,
          'save_us_S2': 27.0,
          'save_us_primary': 79.5,
          'sm': 456,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:20': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6491,
          'save_us_S1m': 255.6,
          'save_us_S2': 53.4,
          'save_us_primary': 116.5,
          'sm': 480,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:21': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6648,
          'save_us_S1m': 245.1,
          'save_us_S2': 34.1,
          'save_us_primary': 116.6,
          'sm': 504,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:22': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6776,
          'save_us_S1m': 231.1,
          'save_us_S2': 33.6,
          'save_us_primary': 115.1,
          'sm': 528,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:23': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6619,
          'save_us_S1m': 252.2,
          'save_us_S2': 46.7,
          'save_us_primary': 123.4,
          'sm': 552,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:24': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6697,
          'save_us_S1m': 235.6,
          'save_us_S2': 43.6,
          'save_us_primary': 121.6,
          'sm': 576,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:25': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.7035,
          'save_us_S1m': 238.8,
          'save_us_S2': 45.7,
          'save_us_primary': 115.1,
          'sm': 600,
          'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:26': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.5838,
          'save_us_S1m': 380.2,
          'save_us_S2': 74.8,
          'save_us_primary': 204.7,
          'sm': 624,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:27': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.5768,
          'save_us_S1m': 402.5,
          'save_us_S2': 83.8,
          'save_us_primary': 212.6,
          'sm': 594,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:29': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6081,
          'save_us_S1m': 393.3,
          'save_us_S2': 81.0,
          'save_us_primary': 200.8,
          'sm': 580,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:30': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6233,
          'save_us_S1m': 352.1,
          'save_us_S2': 80.8,
          'save_us_primary': 195.0,
          'sm': 600,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:32': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6562,
          'save_us_S1m': 373.1,
          'save_us_S2': 53.7,
          'save_us_primary': 180.8,
          'sm': 640,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:35': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.6708,
          'save_us_S1m': 380.6,
          'save_us_S2': 58.0,
          'save_us_primary': 180.6,
          'sm': 560,
          'stock_kernel': 'MultiCtasKvCgaVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:40': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.4716,
          'save_us_S1m': 854.1,
          'save_us_S2': 247.7,
          'save_us_primary': 487.3,
          'sm': 880,
          'stock_kernel': 'VarSeqQ16Kv256PersistentGroupedSwapsAbForGen'},
 '1:45': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
          'ratio': 0.5001,
          'save_us_S1m': 863.3,
          'save_us_S2': 248.7,
          'save_us_primary': 466.4,
          'sm': 900,
          'stock_kernel': 'VarSeqQ16Kv256PersistentGroupedSwapsAbForGen'},
 '1:6': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
         'ratio': 0.8417,
         'save_us_S1m': 33.0,
         'save_us_S2': 0.0,
         'save_us_primary': 15.4,
         'sm': 288,
         'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:7': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
         'ratio': 0.8011,
         'save_us_S1m': 54.0,
         'save_us_S2': 4.8,
         'save_us_primary': 21.4,
         'sm': 378,
         'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '1:9': {'kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen',
         'ratio': 0.77,
         'save_us_S1m': 77.4,
         'save_us_S2': 10.9,
         'save_us_primary': 33.2,
         'sm': 432,
         'stock_kernel': 'MultiCtasKvVarSeqQ16Kv256StaticGroupedSwapsAbForGen'},
 '4:10': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.692,
          'save_us_S1m': 129.0,
          'save_us_S2': 31.6,
          'save_us_primary': 66.1,
          'sm': 300,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:11': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6736,
          'save_us_S1m': 165.6,
          'save_us_S2': 37.4,
          'save_us_primary': 80.3,
          'sm': 286,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:12': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.723,
          'save_us_S1m': 139.5,
          'save_us_S2': 31.9,
          'save_us_primary': 69.8,
          'sm': 288,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:13': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6906,
          'save_us_S1m': 201.3,
          'save_us_S2': 26.7,
          'save_us_primary': 94.1,
          'sm': 286,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:14': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6045,
          'save_us_S1m': 242.2,
          'save_us_S2': 53.0,
          'save_us_primary': 117.9,
          'sm': 448,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:15': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.7171,
          'save_us_S1m': 187.5,
          'save_us_S2': 26.4,
          'save_us_primary': 87.6,
          'sm': 300,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:16': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5681,
          'save_us_S1m': 297.3,
          'save_us_S2': 83.5,
          'save_us_primary': 155.2,
          'sm': 448,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:17': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.7065,
          'save_us_S1m': 259.2,
          'save_us_S2': 24.0,
          'save_us_primary': 111.5,
          'sm': 272,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:18': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.7217,
          'save_us_S1m': 226.6,
          'save_us_S2': 18.1,
          'save_us_primary': 107.7,
          'sm': 288,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:19': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.7037,
          'save_us_S1m': 225.8,
          'save_us_S2': 50.0,
          'save_us_primary': 112.8,
          'sm': 304,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:20': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5313,
          'save_us_S1m': 450.8,
          'save_us_S2': 132.8,
          'save_us_primary': 222.0,
          'sm': 440,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:21': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5755,
          'save_us_S1m': 420.6,
          'save_us_S2': 90.4,
          'save_us_primary': 205.4,
          'sm': 420,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:22': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5804,
          'save_us_S1m': 413.0,
          'save_us_S2': 89.1,
          'save_us_primary': 205.5,
          'sm': 440,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:23': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5908,
          'save_us_S1m': 398.1,
          'save_us_S2': 89.1,
          'save_us_primary': 201.7,
          'sm': 414,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:24': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.593,
          'save_us_S1m': 397.2,
          'save_us_S2': 90.5,
          'save_us_primary': 201.6,
          'sm': 432,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:25': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6536,
          'save_us_S1m': 369.3,
          'save_us_S2': 67.0,
          'save_us_primary': 175.8,
          'sm': 400,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:26': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.4654,
          'save_us_S1m': 690.7,
          'save_us_S2': 195.8,
          'save_us_primary': 378.3,
          'sm': 572,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:27': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.464,
          'save_us_S1m': 719.6,
          'save_us_S2': 207.0,
          'save_us_primary': 376.7,
          'sm': 540,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:29': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.4863,
          'save_us_S1m': 708.8,
          'save_us_S2': 187.9,
          'save_us_primary': 364.5,
          'sm': 580,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:30': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5029,
          'save_us_S1m': 644.7,
          'save_us_S2': 193.3,
          'save_us_primary': 355.3,
          'sm': 600,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:32': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.566,
          'save_us_S1m': 650.1,
          'save_us_S2': 140.6,
          'save_us_primary': 317.2,
          'sm': 448,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:35': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.563,
          'save_us_S1m': 690.2,
          'save_us_S2': 126.7,
          'save_us_primary': 321.5,
          'sm': 560,
          'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '4:40': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.3324,
          'save_us_S1m': 1622.1,
          'save_us_S2': 630.5,
          'save_us_primary': 955.9,
          'sm': 880,
          'stock_kernel': 'VarSeqQ64Kv256PersistentGroupedKeepsAbForGen'},
 '4:45': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.3579,
          'save_us_S1m': 1605.0,
          'save_us_S2': 620.9,
          'save_us_primary': 899.4,
          'sm': 900,
          'stock_kernel': 'VarSeqQ64Kv256PersistentGroupedKeepsAbForGen'},
 '4:9': {'kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen',
         'ratio': 0.738,
         'save_us_S1m': 110.5,
         'save_us_S2': 23.9,
         'save_us_primary': 49.7,
         'sm': 288,
         'stock_kernel': 'MultiCtasKvVarSeqQ64Kv256StaticGroupedKeepsAbForGen'},
 '6:12': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6844,
          'save_us_S1m': 198.0,
          'save_us_S2': 55.8,
          'save_us_primary': 96.3,
          'sm': 288,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:13': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6436,
          'save_us_S1m': 279.2,
          'save_us_S2': 60.7,
          'save_us_primary': 135.9,
          'sm': 286,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:14': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6553,
          'save_us_S1m': 241.9,
          'save_us_S2': 81.0,
          'save_us_primary': 127.2,
          'sm': 280,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:15': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6846,
          'save_us_S1m': 252.8,
          'save_us_S2': 45.5,
          'save_us_primary': 119.2,
          'sm': 300,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:17': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6687,
          'save_us_S1m': 356.5,
          'save_us_S2': 48.7,
          'save_us_primary': 156.5,
          'sm': 272,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:19': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6616,
          'save_us_S1m': 305.4,
          'save_us_S2': 94.9,
          'save_us_primary': 157.8,
          'sm': 304,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:21': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.5878,
          'save_us_S1m': 502.9,
          'save_us_S2': 142.9,
          'save_us_primary': 264.5,
          'sm': 294,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:23': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.6426,
          'save_us_S1m': 475.2,
          'save_us_S2': 113.0,
          'save_us_primary': 221.2,
          'sm': 276,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'},
 '6:25': {'kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen',
          'ratio': 0.681,
          'save_us_S1m': 415.3,
          'save_us_S2': 78.2,
          'save_us_primary': 202.1,
          'sm': 300,
          'stock_kernel': 'MultiCtasKvVarSeqQ128Kv256StaticGroupedKeepsAbForGen'}}
# END GENERATED TABLE


def _norm_table(raw) -> dict:
    out = {}
    for site, by_q in (raw or {}).items():
        for q_len, by_b in (by_q or {}).items():
            for b, sm in (by_b or {}).items():
                if sm is None:
                    continue
                out.setdefault(str(site), {}).setdefault(int(q_len), {})[int(b)] = int(sm)
    return out


def _load():
    if TABLE_ENV:
        with open(TABLE_ENV) as f:
            js = json.load(f)
        geom = dict(TABLE_GEOMETRY)
        geom.update(js.get("geometry") or {})
        return _norm_table(js.get("sites")), geom, js.get("meta") or {}, f"env:{TABLE_ENV}"
    return _norm_table(DEFAULT_TABLE), dict(TABLE_GEOMETRY), dict(DEFAULT_META), TABLE_SOURCE


TABLE, GEOMETRY, META, SOURCE = _load()


def set_table(table: dict, geometry: dict | None = None, meta: dict | None = None, source: str = "set_table") -> None:
    """Replace the table in this process (tests and benches only)."""
    global TABLE, GEOMETRY, META, SOURCE
    TABLE = _norm_table(table)
    GEOMETRY = dict(TABLE_GEOMETRY if geometry is None else geometry)
    META = dict(meta or {})
    SOURCE = source
    for s in (_KEYS_LOGGED, _SKIPS_LOGGED, _PREWARMED, _FIRST_CALL, _HITS):
        s.clear()


def table_sha() -> str:
    blob = json.dumps({"t": {s: {str(q): {str(b): v for b, v in sorted(bb.items())} for q, bb in sorted(qq.items())}
                             for s, qq in sorted(TABLE.items())}, "g": GEOMETRY}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _active_sites():
    return [s for s in sorted(TABLE) if s in SITES]


def num_entries() -> int:
    """Keys that launch with a virtual SM count (keys measured as stock, value 0, do not count)."""
    return sum(1 for s in _active_sites() for bb in TABLE[s].values() for v in bb.values() if v > 0)


def sm_eff_max() -> int:
    """Largest sm_eff any enabled site can launch with (0 when off or empty). Sizes the KV counter blocks."""
    if not ENABLED:
        return 0
    return max([v for s in _active_sites() for bb in TABLE[s].values() for v in bb.values()] or [0])


def entry(site: str, q_len: int, batch: int):
    """sm_eff for (site, q_len, batch), or None (stock). The table lists every measured batch: a value > 0 is the
    virtual SM count, 0 means the bench kept stock there. A batch the bench did not measure (an eager step between two
    graph buckets) takes the value of the largest listed batch below it, so it never inherits an entry across a
    batch that was measured as stock."""
    if not ENABLED or site not in SITES:
        return None
    by_b = TABLE.get(site, {}).get(int(q_len))
    if not by_b:
        return None
    b = int(batch)
    if b in by_b:
        v = by_b[b]
    else:
        lower = [k for k in by_b if k < b]
        v = by_b[max(lower)] if lower else 0
    return v if v > 0 else None


def default_tile_q(q_len: int, heads_per_kv: int) -> int:
    """The tileSizeQ selectGqGenerationKernel starts from (the cost model only lowers it): an upper bound of stepQ."""
    n = int(q_len) * int(heads_per_kv)
    for t in (8, 16, 32, 64):
        if n <= t:
            return t
    return 128


def need_bytes(sm_eff: int, q_len: int, heads_per_kv: int = 16, head_dim: int = 128) -> int:
    """Scratch bytes a launch with mMultiProcessorCount = sm_eff may touch: partial stats and partial O for
    sm_eff x stepQ rows (kernelParams.h:859-863), stepQ bounded by the default tile, partial O at 4 bytes."""
    rows = int(sm_eff) * default_tile_q(q_len, heads_per_kv)
    return rows * (_STATS_BYTES + int(head_dim) * _PARTIAL_O_ELT) + _SLACK


def counter_bytes(batch: int, num_heads: int, sm: int) -> int:
    """getTrtllmGenMultiCtasKvCounterBytes (launcher.cu:56-65): max(batch * heads, sm) rounded up to 8, x 4 bytes."""
    n = max(int(batch) * int(num_heads), int(sm))
    return ((n + 7) // 8) * 8 * 4


# ------------------------------------------------------------------------------------------------ logging
def _rank() -> str:
    for k in ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID", "RANK"):
        v = os.environ.get(k)
        if v not in (None, ""):
            return v
    return "?"


def _log(msg: str) -> None:
    print(f"[M3AX l1-fmha] {msg}", file=sys.stderr, flush=True)


def summary() -> dict:
    return {
        "enabled": ENABLED,
        "sites": sorted(SITES),
        "active_sites": _active_sites(),
        "source": SOURCE,
        "table_sha": table_sha(),
        "entries": num_entries(),
        "sm_eff_max": sm_eff_max(),
        "geometry": GEOMETRY,
        "prewarm": PREWARM,
        "table": {s: {q: dict(sorted(bb.items())) for q, bb in sorted(qq.items())} for s, qq in sorted(TABLE.items())},
    }


_KEYS_LOGGED: set = set()
_SKIPS_LOGGED: set = set()
_PREWARMED: set = set()
_TLS = threading.local()
_ORIG = None
_STATS = {"calls": 0, "overridden": 0, "skipped": 0}


def _activation_line() -> str:
    return (
        f"rank={_rank()} pid={os.getpid()} enabled={int(ENABLED)} sites={','.join(sorted(SITES)) or '-'} "
        f"table={table_sha()} entries={num_entries()} sm_eff_max={sm_eff_max()} prewarm={int(PREWARM)} "
        f"source={SOURCE}"
    )


_log(_activation_line())


# ------------------------------------------------------------------------------------------------ the override
def installed() -> bool:
    try:
        import flashinfer.decode as fd
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(fd.get_device_sm_count, "_m3ax_l1_fmha", False))


def install() -> None:
    """Replace flashinfer.decode.get_device_sm_count by a thread-local-aware wrapper (once per process)."""
    global _ORIG
    import flashinfer.decode as fd

    cur = fd.get_device_sm_count
    if getattr(cur, "_m3ax_l1_fmha", False):
        return
    _ORIG = cur

    def get_device_sm_count(device):
        v = getattr(_TLS, "sm", None)
        return _ORIG(device) if v is None else v

    get_device_sm_count._m3ax_l1_fmha = True
    get_device_sm_count.__wrapped__ = cur
    fd.get_device_sm_count = get_device_sm_count


class override:
    """Context: every get_device_sm_count call flashinfer.decode makes on this thread returns ``sm`` (None = stock)."""

    def __init__(self, sm):
        self.sm = sm
        self.prev = None

    def __enter__(self):
        if self.sm is not None:
            install()
        self.prev = getattr(_TLS, "sm", None)
        _TLS.sm = self.sm
        return self

    def __exit__(self, *exc):
        _TLS.sm = self.prev
        return False


_PHYS_SM: dict = {}


def _physical_sm(device) -> int:
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _PHYS_SM:
        _PHYS_SM[idx] = int(torch.cuda.get_device_properties(idx).multi_processor_count)
    return _PHYS_SM[idx]


def _geometry_ok(q, kv_dtype, kv_heads: int, page_size: int, phys_sm: int) -> bool:
    import torch

    g = GEOMETRY
    return (
        int(q.shape[1]) == int(g.get("num_heads", -1))
        and int(kv_heads) == int(g.get("num_kv_heads", -1))
        and int(q.shape[2]) == int(g.get("head_dim", -1))
        and int(page_size) == int(g.get("page_size", -1))
        and int(phys_sm) == int(g.get("sm_count", -1))
        and q.dtype == torch.float8_e4m3fn
        and kv_dtype == torch.float8_e4m3fn
    )


def _kv_geometry(kwargs):
    kv = kwargs["kv_cache"]
    k = kv[0] if isinstance(kv, (tuple, list)) else kv.select(1, 0)
    layout = kwargs.get("kv_layout", "HND")
    # HND: [pages, kv_heads, page, dim]; NHD: [pages, page, kv_heads, dim]
    if layout == "HND":
        return k, int(k.shape[1]), int(k.shape[2])
    return k, int(k.shape[2]), int(k.shape[1])


def resolve(site: str, batch: int, q_len: int, kwargs: dict):
    """sm_eff for this call, or None (stock): table entry, geometry match, workspace and counters large enough."""
    sm = entry(site, q_len, batch)
    if sm is None:
        return None
    q = kwargs["query"]
    k, kv_heads, page = _kv_geometry(kwargs)
    phys = _physical_sm(q.device)
    if not _geometry_ok(q, k.dtype, kv_heads, page, phys):
        _note_skip(site, q_len, batch, "geometry")
        return None
    heads = int(q.shape[1])
    ws = kwargs["workspace_buffer"]
    ws_bytes = int(ws.numel()) * int(ws.element_size())
    need = need_bytes(sm, q_len, heads // kv_heads, int(q.shape[2]))
    if need > ws_bytes:
        _note_skip(site, q_len, batch, f"workspace {ws_bytes} < {need}")
        return None
    cnt = kwargs.get("multi_ctas_kv_counter_buffer")
    if cnt is None or int(cnt.numel()) * int(cnt.element_size()) < counter_bytes(batch, heads, sm):
        have = None if cnt is None else int(cnt.numel()) * int(cnt.element_size())
        _note_skip(site, q_len, batch, f"counters {have} < {counter_bytes(batch, heads, sm)}")
        return None
    return sm


def _note_skip(site, q_len, batch, why):
    _STATS["skipped"] += 1
    key = (site, int(q_len), int(batch), why.split(" ")[0])
    if key not in _SKIPS_LOGGED:
        _SKIPS_LOGGED.add(key)
        _log(f"skip rank={_rank()} site={site} q_len={q_len} gb={batch} reason={why} (stock launch)")


def _log_key(site, q_len, batch, sm, phys):
    key = (site, int(q_len), int(batch))
    if key in _KEYS_LOGGED:
        return
    _KEYS_LOGGED.add(key)
    m = META.get(f"{q_len}:{batch}") or {}
    _log(
        f"key rank={_rank()} site={site} q_len={q_len} gb={batch} sm_eff={sm} stock_sm={phys} "
        f"kernel={m.get('kernel', '?')} bench_ratio={m.get('ratio', '?')}"
    )


_HITS: dict = {}


def _log_hit(site, q_len, batch, sm):
    """Runtime evidence from eager launches (mixed steps; CUDA-graph replays run no Python): one line at the n-th
    eager use of a key for n = 1, 2, 4, 8, ... (log-rate bounded, as o8's policy-hit line)."""
    key = (site, int(q_len), int(batch))
    n = _HITS.get(key, 0) + 1
    _HITS[key] = n
    if n & (n - 1) == 0:
        _log(f"hit rank={_rank()} site={site} q_len={q_len} gb={batch} sm_eff={sm} n={n}")


# ------------------------------------------------------------------------------------------------ prewarm
def _prewarm(site: str, fn, kwargs: dict, q_len_now: int) -> None:
    """Load every kernel the table can select for this site before serving: one tiny launch per batch 1..max listed
    batch x the site's q_lens x a max_seq_len ladder, each with the table's sm_eff and with the stock count. Reuses
    the call's own workspace and counters (rezeroed after), so nothing is allocated beyond a few small tensors."""
    import time

    import torch

    _PREWARMED.add(site)
    t0 = time.time()
    try:
        q = kwargs["query"]
        k, kv_heads, page = _kv_geometry(kwargs)
        if kwargs.get("kv_layout", "HND") != "HND":
            _log(f"prewarm rank={_rank()} site={site} skipped: kv_layout {kwargs.get('kv_layout')}")
            return
        by_q = TABLE.get(site, {})
        q_lens = sorted(set(by_q) | {int(q_len_now)})
        max_b = max([max(bb) for bb in by_q.values() if bb] or [1])
        dev = q.device
        heads, dim = int(q.shape[1]), int(q.shape[2])
        cnt = kwargs.get("multi_ctas_kv_counter_buffer")
        if cnt is not None:
            # The counter block holds max(heads * max requests, sm) slots (sized by the caller from its request
            # bound): no launch of this process can have more requests than it covers, so neither does the prewarm.
            max_b = max(1, min(max_b, (int(cnt.numel()) * int(cnt.element_size())) // 4 // heads))
        pool = torch.zeros((4, kv_heads, page, dim), dtype=k.dtype, device=dev)
        out_dtype = kwargs["out"].dtype
        shared = bool(kwargs.get("uses_shared_paged_kv_idx", True))
        calls = fails = 0
        variants = set()
        for ql in q_lens:
            for b in range(1, max_b + 1):
                query = torch.zeros((b * ql, heads, dim), dtype=q.dtype, device=dev)
                out = torch.empty((b * ql, heads, dim), dtype=out_dtype, device=dev)
                sm_tab = resolve(site, b, ql, dict(kwargs, query=query))
                for msl in PREWARM_MAX_SEQ_LENS:
                    width = -(-msl // page)
                    if shared:
                        bt = torch.zeros((b, width), dtype=torch.int32, device=dev)
                    else:
                        bt = torch.zeros((b, 2, width), dtype=torch.int32, device=dev)
                        bt[:, 1] = 1
                    seq = torch.full((b,), max(ql, min(msl, 256)), dtype=torch.int32, device=dev)
                    kw = dict(kwargs)
                    kw.update(query=query, kv_cache=(pool, pool), block_tables=bt, seq_lens=seq, max_seq_len=msl,
                              out=out, q_len_per_req=ql, max_q_len=None, cum_seq_lens_q=None)
                    for sm in {None, sm_tab}:
                        try:
                            with override(sm):
                                fn(**kw)
                            calls += 1
                            variants.add((ql, sm is not None))
                        except Exception as exc:  # noqa: BLE001
                            fails += 1
                            if fails <= 3:
                                _log(f"prewarm rank={_rank()} site={site} q_len={ql} b={b} msl={msl} sm={sm} failed: {exc!r}"[:400])
        if cnt is not None:
            cnt.zero_()
        torch.cuda.synchronize(dev)
        PREWARM_STATS[site] = {"q_lens": q_lens, "max_b": max_b, "calls": calls, "failed": fails,
                               "seconds": round(time.time() - t0, 2)}
        _log(
            f"prewarm done rank={_rank()} site={site} q_lens={q_lens} batches=1..{max_b} max_seq_lens={len(PREWARM_MAX_SEQ_LENS)} "
            f"calls={calls} failed={fails} seconds={time.time() - t0:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"prewarm rank={_rank()} site={site} aborted: {exc!r}"[:400])


# ------------------------------------------------------------------------------------------------ the call
_FIRST_CALL: set = set()
PREWARM_STATS: dict = {}


def _log_first_call(site, batch, q_len, kwargs):
    _FIRST_CALL.add(site)
    try:
        import torch

        ws, cnt = kwargs.get("workspace_buffer"), kwargs.get("multi_ctas_kv_counter_buffer")
        q = kwargs["query"]
        _log(
            f"site rank={_rank()} site={site} first call: batch={batch} q_len={q_len} query={tuple(q.shape)} {q.dtype} "
            f"q_len_per_req={kwargs.get('q_len_per_req')} max_q_len={kwargs.get('max_q_len')} "
            f"cum_seq_lens_q={'yes' if kwargs.get('cum_seq_lens_q') is not None else 'no'} "
            f"kv_layout={kwargs.get('kv_layout')} shared_idx={kwargs.get('uses_shared_paged_kv_idx')} "
            f"max_seq_len={kwargs.get('max_seq_len')} "
            f"ws_bytes={None if ws is None else int(ws.numel()) * int(ws.element_size())} "
            f"counter_bytes={None if cnt is None else int(cnt.numel()) * int(cnt.element_size())} "
            f"capturing={int(torch.cuda.is_current_stream_capturing())} table_site={int(site in TABLE)}"
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"site rank={_rank()} site={site} first call (log failed: {exc!r})")


def call(site: str, fn, *, batch: int, q_len: int, kwargs: dict):
    """Run ``fn(**kwargs)`` (flashinfer.decode.trtllm_batch_decode_with_kv_cache) for one decode FMHA call site,
    with the table's sm_eff for (site, q_len, batch) when there is one."""
    if not ENABLED:
        return fn(**kwargs)
    if site not in _FIRST_CALL:
        _log_first_call(site, batch, q_len, kwargs)
    if site not in SITES or site not in TABLE:
        return fn(**kwargs)
    import torch

    _STATS["calls"] += 1
    if PREWARM and site not in _PREWARMED and not torch.cuda.is_current_stream_capturing():
        _prewarm(site, fn, kwargs, q_len)
    sm = resolve(site, batch, q_len, kwargs)
    if sm is None:
        return fn(**kwargs)
    _STATS["overridden"] += 1
    _log_key(site, q_len, batch, sm, _physical_sm(kwargs["query"].device))
    if not torch.cuda.is_current_stream_capturing():
        _log_hit(site, q_len, batch, sm)
    with override(sm):
        return fn(**kwargs)


__all__ = [
    "ENABLED", "SITES", "TABLE", "GEOMETRY", "META", "SOURCE", "PREWARM_MAX_SEQ_LENS",
    "call", "counter_bytes", "default_tile_q", "entry", "install", "installed", "need_bytes", "num_entries",
    "override", "resolve", "set_table", "sm_eff_max", "summary", "table_sha",
]
