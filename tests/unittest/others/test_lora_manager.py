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
"""Unit tests for LoraManager._retain_device_tensors behavior.

Verifies that GPU tensors are not accumulated in _lora_weights when the
PyTorch backend's C++ PeftCacheManager is provided, preventing OOM with
many unique LoRA adapters.
"""

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import torch
from safetensors.torch import save_file

from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.mapping import Mapping


@dataclass
class MockModelConfig:
    """Minimal model config for LoraManager tests."""

    lora_target_modules: list = field(default_factory=lambda: ["attn_q", "attn_k", "attn_v"])
    trtllm_modules_to_hf_modules: dict = field(
        default_factory=lambda: {
            "attn_q": "q_proj",
            "attn_k": "k_proj",
            "attn_v": "v_proj",
        }
    )
    hidden_size: int = 64
    dtype: str = "float16"
    swap_gate_up_proj_lora_b_weight: bool = True


def _create_dummy_hf_lora_adapter(
    adapter_dir: Path, hidden_size: int = 64, rank: int = 8, num_layers: int = 2
):
    """Create a minimal HF-format LoRA adapter on disk."""
    config = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
        "bias": "none",
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
    }
    with open(adapter_dir / "adapter_config.json", "w") as f:
        json.dump(config, f)

    weights = {}
    for layer_idx in range(num_layers):
        for module in ["q_proj", "k_proj", "v_proj"]:
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}"
            weights[f"{prefix}.lora_A.weight"] = torch.randn(rank, hidden_size, dtype=torch.float16)
            weights[f"{prefix}.lora_B.weight"] = torch.randn(hidden_size, rank, dtype=torch.float16)

    save_file(weights, str(adapter_dir / "adapter_model.safetensors"))


def _cpu_cache_bytes(manager) -> int:
    """Total bytes retained in the Python LoraManager's CPU weight store."""
    return sum(t.element_size() * t.nelement()
               for t in manager._cpp_lora_weights.values())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoraManagerRetainDeviceTensors(unittest.TestCase):
    """Tests for the _retain_device_tensors flag that prevents GPU memory leaks."""

    def _create_manager(self, cpp_peft_cache_manager=None):
        mapping = Mapping(world_size=1, rank=0, tp_size=1)
        model_config = MockModelConfig()
        return LoraManager(
            mapping=mapping,
            model_config=model_config,
            cpp_peft_cache_manager=cpp_peft_cache_manager,
        )

    def test_retain_device_tensors_true_when_no_cpp_cache(self):
        """Legacy TRT path: cpp_peft_cache_manager=None retains GPU tensors."""
        manager = self._create_manager(cpp_peft_cache_manager=None)
        self.assertTrue(manager._retain_device_tensors)

    def test_retain_device_tensors_false_when_cpp_cache_provided(self):
        """PyTorch path: cpp_peft_cache_manager provided skips GPU tensor retention."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)
        self.assertFalse(manager._retain_device_tensors)

    def test_lora_weights_empty_with_cpp_cache(self):
        """With cpp_peft_cache_manager, _lora_weights stays empty after loading."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_0"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(adapter_dir)

            model_config = MockModelConfig()
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["test-uid-0"],
            )

        self.assertEqual(len(manager._lora_weights), 0)
        self.assertIn("test-uid-0", manager._cpp_lora_weights)

    def test_lora_weights_populated_without_cpp_cache(self):
        """Without cpp_peft_cache_manager (TRT), _lora_weights has GPU tensors."""
        manager = self._create_manager(cpp_peft_cache_manager=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter_0"
            adapter_dir.mkdir()
            _create_dummy_hf_lora_adapter(adapter_dir)

            model_config = MockModelConfig()
            manager.load_from_hf(
                model_dirs=[str(adapter_dir)],
                model_config=model_config,
                uids=["test-uid-0"],
            )

        self.assertGreater(len(manager._lora_weights), 0)
        self.assertTrue(all(t.is_cuda for t in manager._lora_weights))
        self.assertIn("test-uid-0", manager._lora_weights_pointers_list)

    def test_many_adapters_no_gpu_accumulation(self):
        """Loading many adapters with cpp_cache does not accumulate GPU tensors."""
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)
        model_config = MockModelConfig()

        num_adapters = 20
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_adapters):
                adapter_dir = Path(tmpdir) / f"adapter_{i}"
                adapter_dir.mkdir()
                _create_dummy_hf_lora_adapter(adapter_dir)

                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=[f"uid-{i}"],
                )

        self.assertEqual(len(manager._lora_weights), 0)
        self.assertEqual(len(manager._cpp_lora_weights), num_adapters)

    def test_cpu_lora_weights_grows_with_distinct_adapters(self):
        """_cpp_lora_weights retains one CPU tensor per distinct adapter, unbounded.

        Even though the C++ PeftCacheManager bounds GPU/host *cache* residency via
        max_cpu_loras, the Python LoraManager keeps a CPU copy of every adapter it
        has ever loaded and never evicts. This test documents that host-side growth
        is linear in the number of distinct adapters.
        """
        mock_cache = MagicMock()
        manager = self._create_manager(cpp_peft_cache_manager=mock_cache)
        model_config = MockModelConfig()

        # This test must exercise the PyTorch workflow (C++ PeftCacheManager
        # present), not the legacy TRT path.
        self.assertFalse(manager._retain_device_tensors)

        num_adapters = 20
        bytes_after_first = None
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_adapters):
                adapter_dir = Path(tmpdir) / f"adapter_{i}"
                adapter_dir.mkdir()
                _create_dummy_hf_lora_adapter(adapter_dir)

                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=[f"uid-{i}"],
                )
                total_bytes = _cpu_cache_bytes(manager)
                if i == 0:
                    bytes_after_first = total_bytes
                print(f"[cpu_lora_weights] adapters={i + 1:>2}  "
                      f"total={total_bytes / 1024:.1f} KiB  "
                      f"per_adapter={total_bytes / (i + 1) / 1024:.1f} KiB")

        # PyTorch workflow: no GPU tensors retained in the Python manager ...
        self.assertEqual(len(manager._lora_weights), 0)
        # ... but one CPU tensor retained per distinct adapter ...
        self.assertEqual(len(manager._cpp_lora_weights), num_adapters)
        # ... and total CPU bytes scale linearly (dummy adapters are identical in size).
        self.assertGreater(bytes_after_first, 0)
        self.assertEqual(_cpu_cache_bytes(manager),
                         bytes_after_first * num_adapters)

    def test_cpp_cache_sizes_and_cpu_growth(self):
        """Report C++ host/device cache sizes AND Python CPU-store growth together.

        Builds a *real* C++ PeftCacheManager (not a mock) so we can read the actual
        bounded host/device cache page counts, and uses its wired LoraManager to show
        the (unbounded) Python CPU weight store growing per distinct adapter.
        """
        from tensorrt_llm._torch.pyexecutor.resource_manager import \
            PeftCacheManager as PyPeftCacheManager
        from tensorrt_llm.bindings import DataType
        from tensorrt_llm.bindings import LoraModule
        from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
        from tensorrt_llm.llmapi.llm_args import PeftCacheConfig as \
            LlmApiPeftCacheConfig
        from tensorrt_llm.lora_helper import LoraConfig

        hidden_size = 64
        num_heads = 4
        head_size = hidden_size // num_heads
        num_layers = 2
        mlp_hidden_size = 4 * hidden_size
        target_modules = ["attn_q", "attn_k", "attn_v"]

        # A real C++ ModelConfig with LoRA modules so the cache can be sized.
        model_config_cpp = ModelConfigCpp(
            vocab_size=128,
            num_layers=num_layers,
            num_attention_layers=num_layers,
            num_rnn_layers=0,
            num_heads=num_heads,
            hidden_size=hidden_size,
            data_type=DataType.HALF,
        )
        model_config_cpp.set_num_kv_heads(num_heads)
        model_config_cpp.mlp_hidden_size = mlp_hidden_size
        model_config_cpp.use_lora_plugin = True
        model_config_cpp.max_lora_rank = 8
        model_config_cpp.lora_modules = LoraModule.create_lora_modules(
            lora_module_names=target_modules,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            num_attention_heads=num_heads,
            num_kv_attention_heads=num_heads,
            attention_head_size=head_size,
            tp_size=1,
        )

        # host (max_cpu_loras-like) configured larger than device (max_loras-like).
        peft_cache_config = LlmApiPeftCacheConfig(
            num_host_module_layer=192,
            num_device_module_layer=96,
            optimal_adapter_size=8,
            max_adapter_size=8,
        )
        lora_config = LoraConfig(
            lora_target_modules=target_modules,
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=4,
        )

        pcm = PyPeftCacheManager(
            peft_cache_config=peft_cache_config,
            lora_config=lora_config,
            model_config=model_config_cpp,
        )

        max_host_pages = pcm.impl.max_host_pages
        max_device_pages = pcm.impl.max_device_pages
        print(f"[cpp_cache] max_host_pages={max_host_pages}  "
              f"max_device_pages={max_device_pages}")

        # Both C++ caches are bounded and non-empty, and the host cache is
        # configured larger than the device cache.
        self.assertGreater(max_host_pages, 0)
        self.assertGreater(max_device_pages, 0)
        self.assertGreater(max_host_pages, max_device_pages)

        # The LoraManager wired to the real C++ cache must be on the PyTorch path.
        manager = pcm.get_lora_manager()
        self.assertFalse(manager._retain_device_tensors)

        model_config = MockModelConfig()
        num_adapters = 20
        bytes_after_first = None
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_adapters):
                adapter_dir = Path(tmpdir) / f"adapter_{i}"
                adapter_dir.mkdir()
                _create_dummy_hf_lora_adapter(adapter_dir)

                manager.load_from_hf(
                    model_dirs=[str(adapter_dir)],
                    model_config=model_config,
                    uids=[f"uid-{i}"],
                )
                total_bytes = _cpu_cache_bytes(manager)
                if i == 0:
                    bytes_after_first = total_bytes
                print(f"[lora_caches] adapters={i + 1:>2}  "
                      f"cpp_host_pages={max_host_pages}  "
                      f"cpp_device_pages={max_device_pages}  "
                      f"cpu_store_total={total_bytes / 1024:.1f} KiB  "
                      f"cpu_store_per_adapter={total_bytes / (i + 1) / 1024:.1f} KiB")

        # C++ caches are page-bounded, but the Python CPU store grows one tensor
        # per distinct adapter, unbounded by max_host_pages.
        self.assertEqual(len(manager._lora_weights), 0)
        self.assertEqual(len(manager._cpp_lora_weights), num_adapters)
        self.assertGreater(bytes_after_first, 0)
        self.assertEqual(_cpu_cache_bytes(manager),
                         bytes_after_first * num_adapters)


if __name__ == "__main__":
    unittest.main()
