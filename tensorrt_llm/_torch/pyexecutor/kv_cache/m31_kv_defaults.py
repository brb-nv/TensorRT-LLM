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
"""MiniMax-M3.1 KV cache defaults with DSpark on Blackwell-family GPUs (sm_100 / sm_103 / sm_107).

``arm_minimax_m31`` arms them process-wide before the KV cache managers are built: from the MiniMax-M3.1 targets'
routing (``routing.py``; rather than in a target, so that the target directories, the shipped CuTe DSL AOT objects'
source digests, are not edited) and from the built-in ``MiniMaxM3ForCausalLM`` with DSpark. Each item keeps its own
switch (see its module). Arming sets no environment variable: the shipped AOT objects are keyed on the process's
TRTLLM_M31_DISABLE_* / TRTLLM_M31_ENABLE_* settings (``cute_aot.env_settings``).
"""

from __future__ import annotations

import torch

from . import kv_cache_manager_v2, kv_layerwise_onboard

#: Blackwell-family compute capability major (sm_100 / sm_103 / sm_107): where arm_minimax_m31 arms.
BLACKWELL_MAJOR = 10


def arm_minimax_m31(capability=None) -> None:
    """Arm the MiniMax-M3.1 KV defaults in this process for a Blackwell-family device (``capability``: the
    (major, minor) compute capability, default the current device's): whole-migration host-KV onboarding
    (kv_layerwise_onboard KVLO_OFF) and an unpaired separate draft pool (kv_cache_manager_v2 UNPAIRED_DRAFT_POOL).
    Other devices arm nothing."""
    if capability is None:
        if not torch.cuda.is_available():
            return
        capability = torch.cuda.get_device_capability()
    if tuple(capability)[0] != BLACKWELL_MAJOR:
        return
    kv_layerwise_onboard.arm_m31_layerwise_onboard_off()
    kv_cache_manager_v2.arm_m31_unpaired_draft_pool()
