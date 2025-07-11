import copy
import os
from typing import List, Optional, Tuple

import torch
from transformers import (AutoModel, AutoProcessor, Gemma3Config,
                          PreTrainedModel)
from transformers.modeling_utils import no_init_weights
from transformers.models.gemma3.modeling_gemma3 import Gemma3MultiModalProjector

from ..._utils import nvtx_range
from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


# Make this a runtime lookup rather than a module-wide constant for easier unit testing.
def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class Gemma3InputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer, trust_remote_code):

        self.tokenizer = tokenizer
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=True)
        self.model_config = model_config
        self.device = 'cuda'

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data", {})
        if "image" not in mm_data:
            raise KeyError("Expected image data in multimodal data for Gemma3.")

        images = mm_data["image"]
        if len(images) != 1:
            raise ValueError(
                f"Expected exactly one image for processing, got {len(images)}."
            )
        image = images[0]

        do_rescale = self.processor.image_processor.do_rescale
        if isinstance(image, torch.Tensor):
            do_rescale = False
        processor_output = self.processor(
            text=text_prompt,
            images=image,
            do_rescale=do_rescale,
            return_tensors="pt",
            device=self.device).to(dtype=torch.bfloat16)
        result_dict = {}
        result_dict["prompt"] = inputs["prompt"]
        result_dict["multimodal_data"] = {
            "image": [processor_output["pixel_values"]]
        }
        result_dict["mm_processor_kwargs"] = {}
        for key in ["input_ids", "token_type_ids", "pixel_values"]:
            result_dict["mm_processor_kwargs"][key] = processor_output[key]

        return [result_dict]

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        preprocess_outputs = self._preprocess(inputs)
        input_ids = preprocess_outputs[0]["mm_processor_kwargs"]["input_ids"]
        multimodal_data = {}
        multimodal_data["image"] = {
            "pixel_values":
            preprocess_outputs[0]["mm_processor_kwargs"]["pixel_values"],
        }
        return input_ids[0].to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data
        }


@register_auto_model("Gemma3ForConditionalGeneration")
@register_input_processor(Gemma3InputProcessor, model_type="gemma3")
class Gemma3VLM(PreTrainedModel):

    def __init__(self, model_config: ModelConfig[Gemma3Config]):
        if _is_disagg():
            raise NotImplementedError(
                "Gemma3VLM does not support disaggregated inference yet. Please unset "
                f"the {_MULTIMODAL_ENV_NAME} environment variable, or set it to '0'."
            )

        config = model_config.pretrained_config
        super().__init__(config)

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

        device = torch.device("cuda")
        # Use HF implementations. They should eventually be replaced with TRTLLM counterparts.
        # NOTE: we init the weights after transferring to the `device` since it can take a much
        # longer time to initialize them on the CPU.
        with no_init_weights():
            self.vision_tower = AutoModel.from_config(
                config.vision_config).eval().to(device)
            self.mm_projector = Gemma3MultiModalProjector(config).eval().to(
                device)

        self.post_config()
        self.is_loaded = True

    def load_weights(self, weights):
        llm_weights = filter_weights("language_model", weights)
        self.llm.load_weights(llm_weights)

        vision_tower_weights = filter_weights("vision_tower", weights)
        missing_keys, _ = self.vision_tower.load_state_dict(
            vision_tower_weights)
        if len(missing_keys) > 0:
            raise KeyError(
                "Missing the following keys for the vision tower in the checkpoint: "
                f"[{', '.join(missing_keys)}].")

        projector_weights = filter_weights("multi_modal_projector", weights)
        missing_keys, _ = self.mm_projector.load_state_dict(projector_weights)
        if len(missing_keys) > 0:
            raise KeyError(
                "Missing the following keys for the multi modal projector in the checkpoint: "
                f"[{', '.join(missing_keys)}].")

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

        multimodal_params = kwargs.get("multimodal_params", [])
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]
        assert pixel_values == [] or len(
            pixel_values
        ) == num_context_requests, "Number of multimodal features (if provided) should be equal to number of context requests"

        mm_token_ids = torch.tensor([self.image_token_index
                                     ]).to(input_ids.device)
        mm_token_mask = None
        mm_embeds = []
        if len(pixel_values) > 0:
            # The shape of `image_features` is `[B, T, embed_dim]`.
            image_features = self._get_image_features(
                pixel_values=torch.cat(pixel_values))
            # We need to reshape it to `[B * T, embed_dim]` before passing to `fuse_input_embeds`.
            B, T, embed_dim = image_features.shape
            mm_embeds = [image_features.reshape(B * T, embed_dim).contiguous()]

            # Get token type ids. 0 corresponds to text tokens, 1 corresponds to image tokens.
            mm_token_mask = torch.isin(input_ids, mm_token_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=mm_token_ids)
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            image_token_mask=mm_token_mask,
        )
        return logits

    @nvtx_range("[Vision] process")
    def _get_image_features(self, pixel_values):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            image_features: Tuple[torch.Tensor] = self.vision_tower(
                pixel_values).last_hidden_state
            image_features = self.mm_projector(image_features)
        return image_features
