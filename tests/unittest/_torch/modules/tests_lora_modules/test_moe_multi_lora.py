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
"""Multi-LoRA correctness on MoE models (Qwen1.5-MoE-A2.7B-Chat, O(10) adapters).

Positive coverage: attention and shared-expert MLP targets across O(10)
co-resident adapters. Negative coverage: per-expert MoE targets (moe_*)
are dropped with a warning at LLM construction; the LLM still runs and
attention LoRA is applied. CPU-only filter unit tests live in
tests/unittest/_torch/lora/test_lora.py.
"""

import json
import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file
from utils.llm_data import llm_models_root
from utils.util import skip_gpu_memory_less_than_40gb

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_helper import LoraConfig

_HF_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_HF_SHARED_EXPERT_MODULES = ["gate_proj", "up_proj", "down_proj"]

_TRTLLM_ATTN_MODULES = ["attn_q", "attn_k", "attn_v", "attn_dense"]
# Qwen MoE shared expert is a GatedMLP(is_shared_expert=True) which registers
# SHARED_EXPERT_* LoRA types; using `mlp_*` would cause lora_manager to drop
# the weights with a UserWarning.
_TRTLLM_SHARED_EXPERT_MODULES = [
    "shared_expert_h_to_4h",
    "shared_expert_gate",
    "shared_expert_4h_to_h",
]


def _load_qwen_moe_dims(model_path):
    """Read dimensions from a Qwen-MoE HF config.json."""
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    num_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim", hidden // num_heads)
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    return {
        "hidden": hidden,
        "q_dim": num_heads * head_dim,
        "kv_dim": num_kv_heads * head_dim,
        "moe_intermediate": cfg["moe_intermediate_size"],
        "shared_intermediate": cfg["shared_expert_intermediate_size"],
        "num_layers": cfg["num_hidden_layers"],
        "num_experts": cfg["num_experts"],
    }


def _rand_lora_pair(rank, in_dim, out_dim, dtype, generator):
    """Return (lora_A, lora_B) with small but non-trivial magnitudes."""
    lora_a = (torch.randn(rank, in_dim, dtype=torch.bfloat16, generator=generator) * 0.1).to(dtype)
    lora_b = (torch.randn(out_dim, rank, dtype=torch.bfloat16, generator=generator) * 0.1).to(dtype)
    return lora_a, lora_b


def _write_adapter(output_dir, weights, target_modules, base_model_path, lora_rank):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "base_model_name_or_path": base_model_path,
                "bias": "none",
                "peft_type": "LORA",
                "r": lora_rank,
                "lora_alpha": lora_rank * 2,
                "target_modules": target_modules,
                "task_type": "CAUSAL_LM",
            },
            f,
        )
    save_file(weights, os.path.join(output_dir, "adapter_model.safetensors"))
    return output_dir


def _create_attention_plus_shared_expert_adapter(
    output_dir, base_model_path, dims, *, lora_rank, dtype, seed, include_shared_expert
):
    """Build a Qwen MoE LoRA adapter on attention (+ optionally shared expert)."""
    generator = torch.Generator().manual_seed(seed)
    layer_blocks = {
        "self_attn": {
            "q_proj": (dims["hidden"], dims["q_dim"]),
            "k_proj": (dims["hidden"], dims["kv_dim"]),
            "v_proj": (dims["hidden"], dims["kv_dim"]),
            "o_proj": (dims["q_dim"], dims["hidden"]),
        },
    }
    target_modules = list(_HF_ATTN_MODULES)
    if include_shared_expert:
        layer_blocks["mlp.shared_expert"] = {
            "gate_proj": (dims["hidden"], dims["shared_intermediate"]),
            "up_proj": (dims["hidden"], dims["shared_intermediate"]),
            "down_proj": (dims["shared_intermediate"], dims["hidden"]),
        }
        for m in _HF_SHARED_EXPERT_MODULES:
            if m not in target_modules:
                target_modules.append(m)

    weights = {}
    for layer_idx in range(dims["num_layers"]):
        for block_path, modules in layer_blocks.items():
            for module, (in_dim, out_dim) in modules.items():
                key = f"base_model.model.model.layers.{layer_idx}.{block_path}.{module}"
                lora_a, lora_b = _rand_lora_pair(lora_rank, in_dim, out_dim, dtype, generator)
                weights[f"{key}.lora_A.weight"] = lora_a
                weights[f"{key}.lora_B.weight"] = lora_b

    return _write_adapter(output_dir, weights, target_modules, base_model_path, lora_rank)


def _outputs_differ(lora_out, base_out, tol=1e-6):
    """True if a request's LoRA output differs from its base output (tokens or logprobs)."""
    if list(lora_out.outputs[0].token_ids) != list(base_out.outputs[0].token_ids):
        return True
    lp_lora = lora_out.outputs[0].logprobs
    lp_base = base_out.outputs[0].logprobs
    if lp_lora and lp_base:
        for lp_w, lp_wo in zip(lp_lora, lp_base, strict=True):
            v_w = next(iter(lp_w.values())).logprob
            v_wo = next(iter(lp_wo.values())).logprob
            if abs(v_w - v_wo) > tol:
                return True
    return False


@skip_gpu_memory_less_than_40gb
class TestQwen15MoEMultiLoRA:
    """O(10) co-resident LoRA adapters on Qwen1.5-MoE-A2.7B-Chat."""

    NUM_LORAS = 10
    LORA_RANK = 8
    DTYPE = torch.bfloat16

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen1.5-MoE-A2.7B-Chat"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")
        self.dims = _load_qwen_moe_dims(self.model_path)

    def _build_adapters(self, root, *, include_shared_expert):
        adapter_dirs = []
        for i in range(self.NUM_LORAS):
            adapter_dir = os.path.join(root, f"lora_{i}")
            _create_attention_plus_shared_expert_adapter(
                adapter_dir,
                self.model_path,
                self.dims,
                lora_rank=self.LORA_RANK,
                dtype=self.DTYPE,
                seed=1000 + i,
                include_shared_expert=include_shared_expert,
            )
            adapter_dirs.append(adapter_dir)
        return adapter_dirs

    def _run_with_and_without_lora(self, lora_config, adapter_dirs, prompts):
        assert len(prompts) == self.NUM_LORAS
        with LLM(
            model=self.model_path,
            backend="pytorch",
            lora_config=lora_config,
            tensor_parallel_size=1,
            max_batch_size=self.NUM_LORAS,
            max_num_tokens=512,
        ) as llm:
            sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=0)
            lora_requests = [
                LoRARequest(f"lora-{i}", i, adapter_dirs[i]) for i in range(self.NUM_LORAS)
            ]
            out_lora = llm.generate(prompts, sampling, lora_request=lora_requests)
            out_base = llm.generate(prompts, sampling)
        return out_lora, out_base

    def _assert_each_adapter_changes_output(self, out_lora, out_base):
        unchanged = [
            i
            for i, (lora, base) in enumerate(zip(out_lora, out_base, strict=True))
            if not _outputs_differ(lora, base)
        ]
        assert not unchanged, (
            f"{len(unchanged)}/{self.NUM_LORAS} adapters produced identical outputs to the base "
            f"model (indices {unchanged}); LoRA likely not applied for those requests."
        )

    def _build_prompts(self):
        return [f"Question {i}: What is the capital of France?" for i in range(self.NUM_LORAS)]

    def test_multi_lora_attention(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dirs = self._build_adapters(tmpdir, include_shared_expert=False)
            lora_config = LoraConfig(
                lora_target_modules=list(_TRTLLM_ATTN_MODULES),
                max_lora_rank=self.LORA_RANK,
                max_loras=self.NUM_LORAS,
                max_cpu_loras=self.NUM_LORAS,
            )
            out_lora, out_base = self._run_with_and_without_lora(
                lora_config, adapter_dirs, self._build_prompts()
            )
            self._assert_each_adapter_changes_output(out_lora, out_base)

    def test_multi_lora_attention_and_shared_expert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dirs = self._build_adapters(tmpdir, include_shared_expert=True)
            lora_config = LoraConfig(
                lora_target_modules=(_TRTLLM_ATTN_MODULES + _TRTLLM_SHARED_EXPERT_MODULES),
                max_lora_rank=self.LORA_RANK,
                max_loras=self.NUM_LORAS,
                max_cpu_loras=self.NUM_LORAS,
            )
            out_lora, out_base = self._run_with_and_without_lora(
                lora_config, adapter_dirs, self._build_prompts()
            )
            self._assert_each_adapter_changes_output(out_lora, out_base)


@skip_gpu_memory_less_than_40gb
class TestPerExpertMoELoRADropped:
    """Per-expert MoE LoRA targets are dropped with a warning; LLM still runs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model_path = f"{llm_models_root()}/Qwen1.5-MoE-A2.7B-Chat"
        if not os.path.exists(self.model_path):
            pytest.skip(f"Model not found: {self.model_path}")
        self.dims = _load_qwen_moe_dims(self.model_path)

    def test_per_expert_moe_targets_dropped(self):
        # Mixed targets: attention names are wired (LoraLayer registered),
        # routed-expert names are not. The PyT filter prunes the latter and
        # construction proceeds; the attention LoRA still affects output.
        lora_rank = 8
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = os.path.join(tmpdir, "lora_attn")
            _create_attention_plus_shared_expert_adapter(
                adapter_dir,
                self.model_path,
                self.dims,
                lora_rank=lora_rank,
                dtype=torch.bfloat16,
                seed=42,
                include_shared_expert=False,
            )

            lora_config = LoraConfig(
                lora_target_modules=[
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_dense",
                    "moe_h_to_4h",
                    "moe_4h_to_h",
                    "moe_gate",
                ],
                max_lora_rank=lora_rank,
                max_loras=1,
                max_cpu_loras=1,
            )

            with LLM(
                model=self.model_path,
                backend="pytorch",
                lora_config=lora_config,
                tensor_parallel_size=1,
                max_batch_size=1,
                max_num_tokens=512,
            ) as llm:
                sampling = SamplingParams(max_tokens=10, temperature=0.0, logprobs=0)
                prompts = ["Question: What is the capital of France?"]
                lora_request = [LoRARequest("lora-attn", 1, adapter_dir)]
                out_lora = llm.generate(prompts, sampling, lora_request=lora_request)
                out_base = llm.generate(prompts, sampling)

                assert _outputs_differ(out_lora[0], out_base[0]), (
                    "Attention LoRA did not affect output despite being a supported target."
                )
