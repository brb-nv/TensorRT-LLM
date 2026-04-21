"""A patch for Qwen3 MoE to make it compatible with torch.export and reduce export time."""

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from ...export.interface import BaseExportPatch, ExportPatchRegistry

# Import SiLUActivation for compatibility check
try:
    from transformers.activations import SiLUActivation

    _SILU_TYPES = (nn.SiLU, SiLUActivation)
except ImportError:
    _SILU_TYPES = (nn.SiLU,)


def _is_silu_activation(act_fn) -> bool:
    """Check if activation function is SiLU or equivalent."""
    return isinstance(act_fn, _SILU_TYPES)


def _forward_moe(self: Qwen3MoeSparseMoeBlock, hidden_states: torch.Tensor):
    # check if we can apply the patch
    if any(getattr(mod, "bias", None) is not None for mod in self.experts.modules()):
        raise NotImplementedError(
            "Qwen3MoeSparseMoeBlock forward patch does not support this model configuration: "
            "expert modules have bias. "
            "The original transformers forward uses torch.nonzero() and tensor indexing "
            "which are not compatible with torch.export on meta tensors."
        )

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    # In transformers 5.x, gate returns (router_logits, routing_weights, selected_experts).
    # The routing logic (softmax, topk, normalization) is now inside the gate.
    _, routing_weights, selected_experts = self.gate(hidden_states)

    # In transformers 5.x, self.experts is a fused object with stacked weight tensors
    # (Parameters directly, not modules with .weight).
    # Use torch_moe_fused directly since weights are already stacked.
    gate_up_param = self.experts.gate_up_proj
    gate_up = gate_up_param.weight if hasattr(gate_up_param, "weight") else gate_up_param
    down_param = self.experts.down_proj
    down = down_param.weight if hasattr(down_param, "weight") else down_param

    # HF format: gate_up is [E, 2*I, H] with gate(w1) first, up(w3) second.
    # TRT-LLM format: w3_w1 is [E, 2*I, H] with up(w3) first, gate(w1) second.
    half = gate_up.shape[1] // 2
    w3_w1_stacked = torch.cat([gate_up[:, half:, :], gate_up[:, :half, :]], dim=1)

    final_hidden_states = torch.ops.auto_deploy.torch_moe_fused(
        hidden_states,
        selected_experts,
        routing_weights,
        w3_w1_stacked_weight=w3_w1_stacked,
        w2_stacked_weight=down,
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states


@ExportPatchRegistry.register("hf_qwen3_moe")
class Qwen3MoePatch(BaseExportPatch):
    """Patch for Qwen3 MoE to make it compatible with torch.export and reduce export time.

    This patch replaces the forward method of Qwen3MoeSparseMoeBlock with
    a version that uses the torch_moe custom operator for better export compatibility.
    """

    def _apply_patch(self):
        """Apply the Qwen3 MoE patch."""
        # Store original forward method
        self.original_values["Qwen3MoeSparseMoeBlock.forward"] = Qwen3MoeSparseMoeBlock.forward

        # Apply patch by replacing the forward method
        Qwen3MoeSparseMoeBlock._original_forward = Qwen3MoeSparseMoeBlock.forward  # type: ignore
        Qwen3MoeSparseMoeBlock.forward = _forward_moe  # type: ignore

    def _revert_patch(self):
        """Revert the Qwen3 MoE patch."""
        # Restore original forward method
        Qwen3MoeSparseMoeBlock.forward = self.original_values["Qwen3MoeSparseMoeBlock.forward"]  # type: ignore

        # Clean up the temporary attribute
        if hasattr(Qwen3MoeSparseMoeBlock, "_original_forward"):
            delattr(Qwen3MoeSparseMoeBlock, "_original_forward")
