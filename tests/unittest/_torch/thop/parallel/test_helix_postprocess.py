# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import pytest
import torch
from parameterized import parameterized

import tensorrt_llm


def baseline(gathered_o, gathered_stats, kv_lora_rank, scale):
    """Reference implementation (libtorch)"""
    # [cp_size, num_tokens, num_heads]
    global_max = gathered_stats[..., 0].max(dim=0, keepdim=True)[0]
    # [cp_size, num_tokens, num_heads]
    corrected_max = gathered_stats[..., 0] - global_max
    corrected_max_exp = torch.exp(corrected_max)
    corrected_sum = gathered_stats[..., 1] * corrected_max_exp
    global_sum = corrected_sum.sum(dim=0, keepdim=True)
    correction = (gathered_stats[..., 1] * corrected_max_exp / global_sum).unsqueeze(-1)
    # Cast gathered_o to float32 for computation, then cast output to bf16 at the end
    gathered_o_fp32 = gathered_o.to(torch.float32).view(*correction.shape[:-1], kv_lora_rank)
    corrected_o = gathered_o_fp32 * correction
    # [num_tokens, num_heads * kv_lora_rank] (bf16)
    corrected_o = corrected_o.view(*gathered_o.shape[:-1], -1).sum(dim=0)
    return corrected_o.to(gathered_o.dtype) * scale


def baseline_native(gathered_o, gathered_stats, kv_lora_rank, scale):
    """Reference implementation for native layout (cp_dim=2)
    gathered_o: [num_tokens, num_heads, cp_size, kv_lora_rank]
    gathered_stats: [num_tokens, num_heads, cp_size, 2]
    """
    # [num_tokens, num_heads, cp_size]
    global_max = gathered_stats[..., 0].max(dim=-1, keepdim=True)[0]
    # [num_tokens, num_heads, cp_size]
    corrected_max = gathered_stats[..., 0] - global_max
    corrected_max_exp = torch.exp(corrected_max)
    corrected_sum = gathered_stats[..., 1] * corrected_max_exp
    global_sum = corrected_sum.sum(dim=-1, keepdim=True)
    correction = (gathered_stats[..., 1] * corrected_max_exp / global_sum).unsqueeze(-1)
    # Cast gathered_o to float32 for computation, then cast output to dtype at the end
    gathered_o_fp32 = gathered_o.to(torch.float32)
    corrected_o = gathered_o_fp32 * correction
    # Sum over cp_size dimension (dim=2), result: [num_tokens, num_heads, kv_lora_rank]
    corrected_o = corrected_o.sum(dim=2)
    # Reshape to [num_tokens, num_heads * kv_lora_rank]
    corrected_o = corrected_o.view(corrected_o.shape[0], -1)
    return corrected_o.to(gathered_o.dtype) * scale


class TestHelixPostProcess(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _test_helix_postprocess(self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype):
        """Test helix postprocessing with given parameters"""
        device = torch.device("cuda")

        # Create test tensors
        # gathered_o_init: [cp_size, num_tokens, num_heads, kv_lora_rank]
        gathered_o_init = torch.empty(
            cp_size, num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device
        ).uniform_(-1, 1)

        # gathered_stats: [cp_size, num_tokens, num_heads, 2]
        gathered_stats = torch.empty(
            cp_size, num_tokens, num_heads, 2, dtype=torch.float32, device=device
        )
        gathered_o_max = torch.max(gathered_o_init, dim=-1, keepdim=True)[0]
        gathered_stats[..., 0] = gathered_o_max[..., 0]
        gathered_o_sum = torch.sum(torch.exp(gathered_o_init - gathered_o_max), dim=-1)
        gathered_stats[..., 1] = gathered_o_sum

        gathered_o = gathered_o_init.view(cp_size, num_tokens, num_heads * kv_lora_rank)

        # Call the custom operator
        output = torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, scale)

        # Compute baseline
        expected_output = baseline(gathered_o, gathered_stats, kv_lora_rank, scale)

        # Compare results
        torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-2)

    def _test_helix_postprocess_native(self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype):
        """Test helix postprocessing native (cp_dim=2) with given parameters"""
        device = torch.device("cuda")

        # Create test tensors for native layout (cp_dim=2)
        # gathered_o: [num_tokens, num_heads, cp_size, kv_lora_rank]
        gathered_o = torch.empty(
            num_tokens, num_heads, cp_size, kv_lora_rank, dtype=dtype, device=device
        ).uniform_(-1, 1)

        # gathered_stats: [num_tokens, num_heads, cp_size, 2]
        gathered_stats = torch.empty(
            num_tokens, num_heads, cp_size, 2, dtype=torch.float32, device=device
        )
        gathered_o_max = torch.max(gathered_o, dim=-1, keepdim=True)[0]
        gathered_stats[..., 0] = gathered_o_max[..., 0]
        gathered_o_sum = torch.sum(torch.exp(gathered_o - gathered_o_max), dim=-1)
        gathered_stats[..., 1] = gathered_o_sum

        # Call the custom operator with cp_dim=2
        output = torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, scale, 2)

        # Compute baseline
        expected_output = baseline_native(gathered_o, gathered_stats, kv_lora_rank, scale)

        # Compare results
        torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-2)

    @parameterized.expand(
        [
            # (cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)
            (4, 8, 2, 64, 1.0, torch.float16),
            (8, 16, 4, 128, 0.5, torch.float16),
            (16, 32, 8, 256, 2.0, torch.float16),
            (4, 8, 2, 64, 1.0, torch.bfloat16),
            (8, 16, 4, 128, 0.5, torch.bfloat16),
            (16, 32, 8, 256, 2.0, torch.bfloat16),
        ]
    )
    def test_helix_postprocess_basic(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype
    ):
        """Test basic helix postprocessing functionality"""
        self._test_helix_postprocess(cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)

    @parameterized.expand(
        [
            # (cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)
            (4, 8, 2, 64, 1.0, torch.float16),
            (8, 16, 4, 128, 0.5, torch.float16),
            (16, 32, 8, 256, 2.0, torch.float16),
            (4, 8, 2, 64, 1.0, torch.bfloat16),
            (8, 16, 4, 128, 0.5, torch.bfloat16),
            (16, 32, 8, 256, 2.0, torch.bfloat16),
        ]
    )
    def test_helix_postprocess_native_basic(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype
    ):
        """Test basic helix postprocessing native (cp_dim=2) functionality"""
        self._test_helix_postprocess_native(cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)

    @parameterized.expand(
        [
            # Test edge cases
            (1, 1, 1, 16, 1.0, torch.float16),  # Minimal sizes
            (256, 1, 1, 16, 1.0, torch.float16),  # Max cp_size
            (128, 1, 1, 16, 1.0, torch.float16),  # Single token
            (4, 8, 1, 16, 1.0, torch.float16),  # Single head
            (4, 8, 2, 2048, 1.0, torch.float16),  # Large kv_lora_rank
        ]
    )
    def test_helix_postprocess_edge_cases(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype
    ):
        """Test edge cases with minimal dimensions"""
        self._test_helix_postprocess(cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)

    @parameterized.expand(
        [
            # Test edge cases for native layout
            (1, 1, 1, 16, 1.0, torch.float16),  # Minimal sizes
            (256, 1, 1, 16, 1.0, torch.float16),  # Max cp_size
            (128, 1, 1, 16, 1.0, torch.float16),  # Single token
            (4, 8, 1, 16, 1.0, torch.float16),  # Single head
            (4, 8, 2, 2048, 1.0, torch.float16),  # Large kv_lora_rank
        ]
    )
    def test_helix_postprocess_native_edge_cases(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype
    ):
        """Test edge cases with minimal dimensions for native layout"""
        self._test_helix_postprocess_native(cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype)

    def test_helix_postprocess_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        device = torch.device("cuda")

        # Test with wrong tensor dimensions
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(4, 8, 3, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)
        gathered_stats = torch.randn(4, 8, 2, 1, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

        # Test with wrong data types
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float32, device=device)
        gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

        # Test with non-contiguous tensors
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float16, device=device).transpose(0, 1)
        gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

    def test_helix_postprocess_native_invalid_inputs(self):
        """Test error handling for invalid inputs for native layout"""
        device = torch.device("cuda")

        # Test with wrong cp_dim (only cp_dim=2 is supported)
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 0)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 1)

        # Test with wrong tensor dimensions (3D instead of 4D)
        gathered_o = torch.randn(8, 2, 256, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

        # Test with wrong data types
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float32, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

        # Test with non-contiguous tensors
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device).transpose(0, 1)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

    def test_helix_postprocess_alignment_requirements(self):
        """Test alignment requirements"""
        device = torch.device("cuda")

        # Test with kv_lora_rank that doesn't satisfy alignment requirements
        # For float16 (2 bytes), kv_lora_rank must be multiple of 8 for 16-byte alignment
        # For bfloat16 (2 bytes), kv_lora_rank must be multiple of 8 for 16-byte alignment

        # This should work (kv_lora_rank = 64 is multiple of 8)
        gathered_o = torch.randn(4, 8, 2 * 64, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

        try:
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)
            # Should not raise an error
        except RuntimeError as e:
            pytest.fail(f"Should not raise error for valid alignment: {e}")

        # Test with kv_lora_rank that doesn't satisfy alignment requirements
        gathered_o = torch.randn(4, 8, 4, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(4, 8, 1, 2, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

    def test_helix_postprocess_native_alignment_requirements(self):
        """Test alignment requirements for native layout"""
        device = torch.device("cuda")

        # Test with kv_lora_rank that doesn't satisfy alignment requirements
        # For float16 (2 bytes), kv_lora_rank must be multiple of 8 for 16-byte alignment
        # For bfloat16 (2 bytes), kv_lora_rank must be multiple of 8 for 16-byte alignment

        # This should work (kv_lora_rank = 64 is multiple of 8)
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        try:
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)
            # Should not raise an error
        except RuntimeError as e:
            pytest.fail(f"Should not raise error for valid alignment: {e}")

        # Test with kv_lora_rank that doesn't satisfy alignment requirements
        gathered_o = torch.randn(8, 1, 4, 4, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 1, 4, 2, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

    def test_helix_postprocess_large_inputs(self):
        """Test with larger inputs to ensure performance and correctness"""
        self._test_helix_postprocess(16, 16, 64, 512, 1.0, torch.float16)
        self._test_helix_postprocess(16, 16, 64, 512, 1.0, torch.bfloat16)

    def test_helix_postprocess_native_large_inputs(self):
        """Test native layout with larger inputs to ensure performance and correctness"""
        self._test_helix_postprocess_native(16, 16, 64, 512, 1.0, torch.float16)
        self._test_helix_postprocess_native(16, 16, 64, 512, 1.0, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
