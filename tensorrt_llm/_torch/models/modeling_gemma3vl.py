import copy
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModel, AutoProcessor, Gemma3Config,
                          PretrainedConfig, PreTrainedModel)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...llmapi.utils import download_hf_model
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model


class Gemma3InputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer, trust_remote_code):

        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model_config = model_config
        self.device = 'cuda'

        # Determine the actual local path for model files
        if os.path.isdir(model_path):
            local_model_path = model_path
        else:
            local_model_path = download_hf_model(model_path)

        # Partially load the model to reduce memory usage(Vision tower and multi-modal projector)
        hf_model_config = AutoConfig.from_pretrained(local_model_path)
        self.dtype = hf_model_config.text_config.torch_dtype
        module_dict = nn.ModuleDict({
            "vision_tower":
            AutoModel.from_config(hf_model_config.vision_config),
            "multi_modal_projector":
            Gemma3MultiModalProjector(hf_model_config)
        })
        missing_keys, _ = load_sharded_checkpoint(module_dict,
                                                  local_model_path,
                                                  strict=False)
        assert len(missing_keys) == 0, f"Missing keys: {missing_keys}"
        hf_vision_tower = module_dict["vision_tower"].to(self.dtype).to(
            self.device)
        hf_mm_projector = module_dict["multi_modal_projector"].to(
            self.dtype).to(self.device)

        # Use HF vision tower. To be replaced with TRTLLM vision tower.
        self.vision_tower = hf_vision_tower

        # Use HF multi-modal projector
        self.mm_projector = hf_mm_projector

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        assert 'image' in mm_data
        processor_output = self.processor(text=text_prompt,
                                          images=mm_data["image"][0],
                                          return_dict=True,
                                          return_tensors="pt",
                                          device=self.device).to(
                                              'cuda', dtype=torch.bfloat16)
        result_dict = {}
        result_dict["prompt"] = inputs["prompt"]
        result_dict["multimodal_data"] = {
            "image": [processor_output["pixel_values"]]
        }
        result_dict["mm_processor_kwargs"] = {}
        for key in ["input_ids", "token_type_ids", "pixel_values"]:
            result_dict["mm_processor_kwargs"][key] = processor_output[key]

        return [result_dict]

    @nvtx_range("[Vision] process")
    def _process(self, pixel_values):
        image_features: Tuple[torch.Tensor] = self.vision_tower(
            pixel_values).last_hidden_state
        image_features = self.mm_projector(image_features)
        return image_features

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        preprocess_outputs = self._preprocess(inputs)
        pixel_values = preprocess_outputs[0]["mm_processor_kwargs"][
            "pixel_values"]
        input_ids = preprocess_outputs[0]["mm_processor_kwargs"]["input_ids"]
        mm_features = self._process(pixel_values)
        return input_ids[0].to(torch.int32).tolist(), {
            "mm_embedding": mm_features
        }


def update_causal_mask(
    attention_mask,
    token_type_ids,
    target_length,
    cache_position,
    input_tensor,
):
    # (Pdb) attention_mask
    # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # (Pdb) attention_mask.shape
    # torch.Size([1, 281])
    # (Pdb) token_type_ids
    # tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #         1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # (Pdb) token_type_ids.shape
    # torch.Size([1, 281])
    # (Pdb) cache_position
    # tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
    #         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
    #         28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
    #         42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
    #         56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
    #         70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
    #         84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
    #         98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    #         112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    #         126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    #         140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    #         154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    #         168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
    #         182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
    #         196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    #         210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    #         224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
    #         238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
    #         252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
    #         266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
    #         280])
    # (Pdb) cache_position.shape
    # torch.Size([281])
    # (Pdb) input_tensor
    # tensor([[     2,    105,   2364,    107,   3048,    659,    496,  11045,  16326,
    #         236761,    110, 255999, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144, 262144,
    #         262144, 262144, 262144, 262144, 262144, 262144, 262144, 256000,    108,
    #         10936,    563,    506,   5866,   8101, 236881,    106,    107,    105,
    #         4368,    107]])
    # (Pdb) input_tensor.shape
    # torch.Size([1, 281])

    sequence_length = input_tensor.shape[-1]
    causal_mask = torch.arange(
        target_length,
        device=cache_position.device) <= cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(1, 1, -1, -1)

    # Apply a bidirectional mask for image tokens of a given image. If there are image tokens from multiple images,
    # tokens from different images will not attend to each other.
    if token_type_ids is not None:
        token_type_mask = token_type_ids.unsqueeze(
            1) == token_type_ids.unsqueeze(2)
        token_type_mask[token_type_ids ==
                        0] = False  # if text token, do not change anything.
        token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device,
                                                          dtype=torch.bool)
        causal_mask = causal_mask.clone()
        causal_mask[:, :, :, :
                    sequence_length] = causal_mask[:, :, :, :
                                                   sequence_length].masked_fill(
                                                       token_type_mask, True)

    if attention_mask is not None:
        causal_mask = causal_mask.clone(
        )  # copy to contiguous memory for in-place edit.
        mask_length = attention_mask.shape[-1]

        # Then apply padding mask (will mask pad tokens).
        padding_mask = causal_mask[:, :, :, :
                                   mask_length] + attention_mask[:, None,
                                                                 None, :].to(
                                                                     causal_mask
                                                                     .device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :
                    mask_length] = causal_mask[:, :, :, :
                                               mask_length].masked_fill(
                                                   padding_mask, False)

    return causal_mask


@register_auto_model("Gemma3ForConditionalGeneration")
@register_input_processor(Gemma3InputProcessor, model_type="gemma3")
class Gemma3Model(PreTrainedModel):
    config_class = Gemma3Config

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs) -> None:
        config = model_config.pretrained_config
        super().__init__(config)
        if hasattr(self, "llm"):
            return

        self.image_token_index = config.image_token_index

        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = model_config.pretrained_config.text_config

        llm_model_config.pretrained_config.torch_dtype = torch.bfloat16
        self.llm = Gemma3ForCausalLM(llm_model_config)

        self.model_config = model_config
        self.vocab_size = config.text_config.vocab_size
        self.model_dtype = getattr(config.text_config, "torch_dtype",
                                   torch.float16)
        logger.info(f"[Gemma3Model::__init__]{self.dtype=} {self.model_dtype=}")

        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):

        weights = filter_weights("language_model", weights)
        self.llm.load_weights(weights)

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        num_context_requests, num_generation_requests = attn_metadata.num_contexts, attn_metadata.num_generations
        logger.debug(
            f"[Gemma3Model::forward]{num_context_requests=}, {num_generation_requests=}"
        )

        mm_embed = kwargs.get("multi_modal_data", [])
        assert mm_embed == [] or len(
            mm_embed
        ) == num_context_requests, "Number of multimodal features (if provided) should be equal to number of context requests"

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embed,
            mm_token_ids=torch.tensor([self.image_token_index
                                       ]).to(input_ids.device))
        if len(mm_embeds) != 0:
            token_type_ids = torch.zeros_like(input_ids,
                                              device=input_ids.device)
            image_token_mask = (input_ids == self.image_token_index).to(
                device=input_ids.device, dtype=torch.bool)
            token_type_ids[image_token_mask] = 1
            attention_mask = update_causal_mask(
                attention_mask=torch.ones(input_ids.shape,
                                          device=input_ids.device),
                token_type_ids=token_type_ids,
                target_length=input_ids.shape[-1],
                cache_position=torch.arange(input_ids.shape[-1],
                                            device=input_ids.device),
                input_tensor=input_ids)
        logits = self.llm.forward(attn_metadata, input_ids, position_ids,
                                  inputs_embeds, return_context_logits)
        return logits


AutoModel.register(Gemma3Config, Gemma3Model)
