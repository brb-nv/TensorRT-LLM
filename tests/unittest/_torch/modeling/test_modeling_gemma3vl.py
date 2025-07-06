import unittest
from copy import deepcopy
from dataclasses import dataclass

import torch
from parameterized import parameterized
from transformers import Gemma3Config
from transformers import \
    Gemma3ForConditionalGeneration as HFGemma3ForConditionalGeneration
from transformers.cache_utils import HybridCache

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma3vl import (Gemma3Model,
                                                          update_causal_mask)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping

GEMMA3_27B_MINI_CONFIG = {
    "architectures": ["Gemma3ForConditionalGeneration"],
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "eos_token_id": [1, 106],
    "image_token_index": 262144,
    "initializer_range": 0.02,
    "mm_tokens_per_image": 256,
    "model_type": "gemma3",
    "text_config": {
        "head_dim": 128,
        "hidden_size": 5376,
        "intermediate_size": 21504,
        "model_type": "gemma3_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 6,  # Modified for testing.
        "num_key_value_heads": 16,
        "query_pre_attn_scalar": 168,
        "rope_scaling": {
            "factor": 8.0,
            "rope_type": "linear"
        },
        "sliding_window": 4  # Modified for testing.
    },
    "torch_dtype": "bfloat16",
    "transformers_version": "4.50.0.dev0",
    "vision_config": {
        "hidden_size": 1152,
        "image_size": 896,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
        "vision_use_head": False
    }
}


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


class TestGemma3(unittest.TestCase):

    def get_kv_cache_manager(self, dtype: torch.dtype, config: Gemma3Config,
                             tokens_per_block: int, max_seq_len: int,
                             batch_size: int, num_blocks: int):
        if dtype == torch.half:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
        elif dtype == torch.bfloat16:
            kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
        else:
            raise ValueError("Invalid dtype")

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                        enable_partial_reuse=False,
                                        copy_on_partial_reuse=False,
                                        max_tokens=num_blocks *
                                        tokens_per_block)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        return kv_cache_manager

    # TODO: Add sanity test once functionality is verified.

    @parameterized.expand([
        Scenario(backend="TRTLLM"),
        Scenario(backend="VANILLA"),
        Scenario(backend="FLASHINFER"),
    ], lambda testcase_func, param_num, param:
                          f"{testcase_func.__name__}[{param.args[0]}]")
    @torch.no_grad()
    def test_gemma3_allclose_to_hf(self, scenario: Scenario) -> None:
        """
        Compare output to HF.
        """
        backend = scenario.backend
        metadata_cls = get_attention_backend(backend).Metadata

        torch.random.manual_seed(0)
        config_dict = deepcopy(GEMMA3_27B_MINI_CONFIG)

        ####################################################################################
        # from PIL import Image
        # import requests
        # from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        # model_dir = "/home/bbuddharaju/scratch/random/hf_models/gemma-3-27b-it/"
        # model = Gemma3ForConditionalGeneration.from_pretrained(model_dir)

        # processor = AutoProcessor.from_pretrained(model_dir)
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {"type": "text", "text": "You are a helpful assistant."}
        #         ]
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        #             {"type": "text", "text": "Where is the cat standing?"},
        #         ]
        #     },
        # ]
        # inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
        # generate_ids = model.generate(**inputs)
        # outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print(outputs)
        ####################################################################################

        gemma3_config = Gemma3Config.from_dict(config_dict)
        dtype = gemma3_config.torch_dtype
        device = torch.device('cuda')

        num_blocks = 1
        tokens_per_block = 128
        max_seq_len = num_blocks * tokens_per_block
        batch_size = 1

        hf_gemma3 = HFGemma3ForConditionalGeneration(gemma3_config).to(
            dtype).to(device).eval()
        hf_cache = HybridCache(config=gemma3_config.text_config,
                               max_batch_size=batch_size,
                               max_cache_len=10,
                               device=device,
                               dtype=dtype)

        model_config = ModelConfig(pretrained_config=gemma3_config,
                                   attn_backend=backend)
        gemma3 = Gemma3Model(model_config).to(dtype).to(device)
        gemma3.load_weights(hf_gemma3.state_dict())

        kv_cache_manager = self.get_kv_cache_manager(
            dtype=dtype,
            config=gemma3_config.text_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        # Context phase.
        input_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
                                 dtype=torch.int32,
                                 device=device)
        num_cached_tokens_per_seq = [0]
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        position_ids = [torch.arange(0, input_ids.size(-1), dtype=torch.int32)]
        position_ids = torch.cat(position_ids).unsqueeze(0).cuda()

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = gemma3.llm.forward(input_ids=input_ids,
                                        position_ids=position_ids,
                                        attn_metadata=attn_metadata)
            ref = hf_gemma3.language_model.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                past_key_values=hf_cache,
                use_cache=True)

            print(
                "[TestGemma3::test_gemma3_allclose_to_hf] max prefill diff: ",
                torch.max(torch.abs(logits - ref.logits[:, -1].float())).item())
            print(
                "[TestGemma3::test_gemma3_allclose_to_hf] mean prefill diff: ",
                torch.mean(torch.abs(logits -
                                     ref.logits[:, -1].float())).item())
            torch.testing.assert_close(logits,
                                       ref.logits[:, -1].float(),
                                       atol=0.1,
                                       rtol=0.1)

        # Generation phase.
        gen_input_ids = torch.tensor([900], dtype=torch.int, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        gen_position_ids = [
            torch.arange(input_ids.size(-1),
                         input_ids.size(-1) + gen_input_ids.size(-1))
        ]
        gen_position_ids = torch.cat(gen_position_ids).unsqueeze(0).cuda()
        with torch.inference_mode():
            attn_metadata.prepare()
            logits = gemma3.llm.forward(input_ids=gen_input_ids,
                                        position_ids=gen_position_ids,
                                        attn_metadata=attn_metadata)
            ref = hf_gemma3.language_model.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=hf_cache,
                use_cache=True,
                cache_position=torch.IntTensor([input_ids.size(-1)]).to(device),
                last_cache_position=input_ids.size(-1) + 1)
            print(
                "[TestGemma3::test_gemma3_allclose_to_hf] max gen diff: ",
                torch.max(torch.abs(logits - ref.logits[:, -1].float())).item())
            print(
                "[TestGemma3::test_gemma3_allclose_to_hf] mean gen diff: ",
                torch.mean(torch.abs(logits -
                                     ref.logits[:, -1].float())).item())
            torch.testing.assert_close(logits,
                                       ref.logits[:, -1].float(),
                                       atol=0.1,
                                       rtol=0.1)

        kv_cache_manager.shutdown()

    def test_gemma3_compare_mask(self) -> None:
        """
      Compare the mask generated by the model with the mask generated by the HF model.
      """
        image_token_index = 262144
        device = torch.device('cuda')
        input_ids = torch.IntTensor([[
            100, 200, image_token_index, image_token_index, image_token_index,
            image_token_index, 700, 800
        ]]).to(device=device)
        token_type_ids = torch.IntTensor([[0, 0, 1, 1, 2, 2, 0,
                                           0]]).to(device=device)
        print("[TestGemma3::test_gemma3_compare_mask] token_type_ids: \n",
              token_type_ids)
        cache_position = torch.arange(input_ids.shape[-1], device=device)
        attention_mask = update_causal_mask(attention_mask=torch.ones_like(
            input_ids, device=device),
                                            token_type_ids=token_type_ids,
                                            target_length=input_ids.shape[-1],
                                            cache_position=cache_position,
                                            input_tensor=input_ids)
        print("[TestGemma3::test_gemma3_compare_mask] attention_mask: \n",
              attention_mask)
        # Image1's tokens don't attend to image2's tokens. Image2's tokens do attend to image1's tokens because of causality.
        expected_attention_mask = torch.tensor(
            [[[[True, False, False, False, False, False, False, False],
               [True, True, False, False, False, False, False, False],
               [True, True, True, True, False, False, False, False],
               [True, True, True, True, False, False, False, False],
               [True, True, True, True, True, True, False, False],
               [True, True, True, True, True, True, False, False],
               [True, True, True, True, True, True, True, False],
               [True, True, True, True, True, True, True, True]]]],
            device=device)
        torch.testing.assert_close(attention_mask, expected_attention_mask)
