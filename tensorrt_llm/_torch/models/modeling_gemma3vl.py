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


def get_gemma3_causal_mask(
    input_ids: torch.Tensor,
    image_token_index: int,
    sliding_window: Optional[int] = None,
):
    print("[get_gemma3_causal_mask] input_ids: ", input_ids)
    assert input_ids.ndim == 1, "input_ids should be a 1D tensor."
    # Get token type ids. 0 corresponds to text tokens, 1 corresponds to image tokens.
    token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)
    image_token_mask = (input_ids == image_token_index).to(
        device=input_ids.device, dtype=torch.bool)
    token_type_ids[image_token_mask] = 1

    sequence_length = input_ids.shape[-1]
    # TODO: Use causal when sliding_window is larger than sequence_length.
    if sliding_window is None:
        causal_mask = torch.arange(
            sequence_length,
            device=input_ids.device).unsqueeze(0) <= torch.arange(
                sequence_length, device=input_ids.device).unsqueeze(1)
    else:
        attention_mask_1 = torch.arange(
            sequence_length,
            device=input_ids.device).unsqueeze(0) <= torch.arange(
                sequence_length, device=input_ids.device).unsqueeze(1)
        attention_mask_2 = torch.arange(
            sequence_length,
            device=input_ids.device).unsqueeze(0) > torch.arange(
                sequence_length,
                device=input_ids.device).unsqueeze(1) - sliding_window
        causal_mask = attention_mask_1 & attention_mask_2

    # Apply a bidirectional mask for image tokens.
    if token_type_ids is not None:
        token_type_mask = token_type_ids.unsqueeze(
            0) == token_type_ids.unsqueeze(1)
        # If text token, do not change anything.
        token_type_mask[token_type_ids == 0] = False
        causal_mask = causal_mask.masked_fill(token_type_mask, True)
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
        self.sliding_window = config.text_config.sliding_window
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

        # Currently, we supply global and local attention masks to the model only when
        # there are image tokens - specifically in the context phase.
        global_attention_mask = None
        local_attention_mask = None
        if len(mm_embed) != 0:
            # Request has image tokens.
            global_attention_mask = get_gemma3_causal_mask(
                input_ids=input_ids, image_token_index=self.image_token_index)
            local_attention_mask = get_gemma3_causal_mask(
                input_ids=input_ids,
                image_token_index=self.image_token_index,
                sliding_window=self.sliding_window,
            )
        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embed,
            mm_token_ids=torch.tensor([self.image_token_index
                                       ]).to(input_ids.device))
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            global_attention_mask_data=global_attention_mask,
            local_attention_mask_data=local_attention_mask,
        )
        return logits


AutoModel.register(Gemma3Config, Gemma3Model)
