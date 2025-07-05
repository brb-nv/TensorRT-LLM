import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers.cache_utils import HybridCache
from transformers import AutoConfig, AutoModel, Gemma3Config
from transformers.modeling_utils import load_sharded_checkpoint

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma3 import Gemma3ForCausalLM
from tensorrt_llm._torch.models.modeling_siglip import SiglipVisionModel
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# This is copied from https://huggingface.co/google/gemma-3-27b-it/blob/main/config.json.
GEMMA3_27B_MINI_VISION_CONFIG = {
    "hidden_size": 1152,
    "image_size": 896,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 1,    # Modified for testing.
    "patch_size": 14,
    "vision_use_head": False
}


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


class TestGemma3VisionModel(unittest.TestCase):

    @parameterized.expand([
        Scenario(backend="TRTLLM"),
        Scenario(backend="VANILLA"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_gemma3_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output of TRTLLM vision tower to HF.
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)

        local_model_path = "/home/bbuddharaju/scratch/random/hf_models/gemma-3-27b-it/"
        hf_model_config = AutoConfig.from_pretrained(local_model_path)
        #################################################################
        # HARDCODING.
        hf_model_config.vision_config.torch_dtype = torch.bfloat16
        hf_model_config.vision_config.num_hidden_layers = 1
        #################################################################
        dtype = hf_model_config.vision_config.torch_dtype
        device = torch.device('cuda')
        assert dtype == torch.bfloat16
        module_dict = torch.nn.ModuleDict({
            "vision_tower":
            AutoModel.from_config(hf_model_config.vision_config),
        })
        missing_keys, _ = load_sharded_checkpoint(module_dict,
                                                  local_model_path,
                                                  strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        hf_vision_tower = module_dict["vision_tower"].to(dtype).to(device).eval()

        vision_model_config = ModelConfig(pretrained_config=hf_model_config.vision_config, attn_backend="TRTLLM")
        trtllm_vision_tower = SiglipVisionModel(vision_model_config).to(device).to(dtype)
        trtllm_vision_tower.load_weights(hf_vision_tower.state_dict())

        pixel_values = torch.rand(1, 3, 896, 896, dtype=torch.bfloat16, device=device)

        with torch.inference_mode():
            attn_metadata = trtllm_vision_tower.prepare_attn_metadata(pixel_values.shape[0])
            trtllm_image_features = trtllm_vision_tower(pixel_values, attn_metadata=attn_metadata)
            hf_image_features = hf_vision_tower(pixel_values)

            print(f"[test_gemma3_allclose_to_hf] trtllm_image_features: {trtllm_image_features[-1].shape} \n trtllm_image_features: {trtllm_image_features[-1]}")
            print(f"[test_gemma3_allclose_to_hf] hf_image_features: {hf_image_features.last_hidden_state.shape} \n hf_image_features: {hf_image_features.last_hidden_state}")
            assert trtllm_image_features[-1].shape == hf_image_features.last_hidden_state.shape
            print("[test_gemma3_allclose_to_hf] max diff:", (trtllm_image_features[-1] - hf_image_features.last_hidden_state).abs().max())
            print("[test_gemma3_allclose_to_hf] mean diff:", (trtllm_image_features[-1] - hf_image_features.last_hidden_state).abs().mean())
            torch.testing.assert_close(trtllm_image_features[-1],
                                       hf_image_features.last_hidden_state,
                                       atol=0.1,
                                       rtol=0.1)
            import pdb; pdb.set_trace()
