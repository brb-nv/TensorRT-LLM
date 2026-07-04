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
"""Shared routed-expert MoE LoRA discovery helpers.

The bits that are common across MoE backends (Cutlass and TRTLLM-gen): building
the discovery-only ``MoeLoraLayer`` marker and detecting whether a layer's
``lora_params`` carries a routed-expert MoE LoRA delta. The actual LoRA math is
backend specific -- Cutlass fuses it into ``torch.ops.trtllm.fused_moe``, while
TRTLLM-gen builds BGMV deltas (see moe_lora_delta.py) -- so the per-backend
tensor extraction is NOT shared here.
"""

from typing import Dict, List, Optional

from ...peft.lora.layer import MOE_LORA_MODULE_NAMES, LoraModuleType, MoeLoraLayer
from ...peft.lora.validation import has_moe_lora_targets

__all__ = [
    "has_moe_lora_targets",
    "make_moe_lora_marker",
    "moe_lora_active",
]


def make_moe_lora_marker(model_config, hidden_size: int,
                         intermediate_size: int) -> Optional[MoeLoraLayer]:
    """Build a discovery-only ``MoeLoraLayer`` marker iff this MoE layer is in the
    routed-expert LoRA target-module set, else ``None``.

    The marker is not a compute submodule: the actual LoRA application is fused
    into the backend kernel path. It exists so ``CudaGraphLoraManager`` and the
    target-module validator can find this MoE layer via ``isinstance(child,
    LoraLayer)`` traversal and read its ``lora_module_types`` /
    ``output_hidden_sizes`` when building slot tables.

    ``output_hidden_sizes`` are the per-token outputs of the LoRA-side GEMM (not
    per-expert weight shapes): MOE_H_TO_4H / MOE_GATE produce ``intermediate_size``,
    MOE_4H_TO_H produces ``hidden_size``.
    """
    lora_config = getattr(model_config, "lora_config", None)
    if lora_config is None:
        return None
    # Normalize to lowercase to match has_moe_lora_targets (which lowercases
    # before comparing), so a mixed-case config marks the layer consistently.
    targets = {
        name.lower()
        for name in (getattr(lora_config, "lora_target_modules", []) or [])
    }
    active_modules: List[LoraModuleType] = []
    active_out_sizes: List[int] = []
    for name in MOE_LORA_MODULE_NAMES:
        if name not in targets:
            continue
        module_type = LoraModuleType.from_string(name)
        if name == "moe_4h_to_h":
            active_out_sizes.append(hidden_size)
        else:
            active_out_sizes.append(intermediate_size)
        active_modules.append(module_type)
    if not active_modules:
        return None
    return MoeLoraLayer(active_modules, active_out_sizes)


def moe_lora_active(layer_idx: Optional[int],
                    lora_params: Optional[Dict]) -> bool:
    """Return True when ``lora_params`` carries a routed-expert MoE LoRA delta for
    the given layer, meaning the backend should fuse/apply a LoRA delta.

    Handles both the eager per-layer dict and the CUDA-graph slot-indexed
    ``cuda_graph_params`` layer map (mirrors the extraction paths).
    """
    if not lora_params or layer_idx is None:
        return False
    # CUDA-graph slot-indexed mode carries MoE LoRA in cuda_graph_params rather
    # than a per-layer eager dict, so consult the graph layer map.
    if lora_params.get("use_cuda_graph_mode", False):
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return False
        layer_module2key = getattr(cuda_graph_params, "layer_module2key", {})
        return any((layer_idx, int(LoraModuleType.from_string(name))) in layer_module2key
                   for name in MOE_LORA_MODULE_NAMES)
    layer_params = lora_params.get(layer_idx, {})
    if not layer_params:
        return False
    return any(
        int(LoraModuleType.from_string(name)) in layer_params
        for name in MOE_LORA_MODULE_NAMES)
