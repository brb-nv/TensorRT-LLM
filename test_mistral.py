import json

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp
from tensorrt_llm.runtime.session import Session, TensorInfo

torch.cuda.set_device(0)
stream = torch.cuda.Stream(torch.cuda.current_device())
torch.cuda.set_stream(stream)

encoder_path = "/home/bbuddharaju/scratch/TensorRT-LLM/mistral_mm_eng/vision"
with open(encoder_path + "/config.json", "r") as f:
    config = json.load(f)
    precision = config["builder_config"]["precision"]
with open(encoder_path + "/visual_encoder.engine", "rb") as f:
    engine_buffer = f.read()
encoder_session = Session.from_serialized_engine(engine_buffer)

# url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
# url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
url = 'https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png'
raw_image = Image.open(requests.get(url, stream=True).raw)

model_path = "/home/bbuddharaju/scratch/random/hf_models/Mistral-Small-3.1-24B-Instruct-2503/"
processor = AutoProcessor.from_pretrained(model_path)
inputs = processor(text="<s>[INST][IMG]What is the image?[/INST]",
                   images=[raw_image],
                   return_tensors="pt").to(str_dtype_to_torch(precision))

dtype = str_dtype_to_torch(precision)
d_min = torch.finfo(dtype).min
pixel_values = torch.full((1, 3, 1540, 1540),
                          fill_value=0,
                          dtype=dtype,
                          device="cuda")  # image_size from config -> 1540.
attention_mask = torch.full((1, 110, 110),
                            fill_value=d_min,
                            dtype=dtype,
                            device="cuda")  # patch_size from config -> 14.

_pixel_values = inputs["pixel_values"].to(device="cuda", dtype=dtype)
h, w = _pixel_values.shape[-2:]
pixel_values[..., :h, :w] = _pixel_values
attention_mask[..., :h // 14, :w // 14] = 0

# model_runner_input ---> pixel_values
# other_vision_inputs ---> attention_mask

visual_features = {"input": pixel_values, "attention_mask": attention_mask}
tensor_info = [
    TensorInfo("input", str_dtype_to_trt(precision), pixel_values.shape),
    TensorInfo("attention_mask", str_dtype_to_trt(precision),
               attention_mask.shape),
]
output_info = encoder_session.infer_shapes(tensor_info)
encoder_session.set_shapes(visual_features)
visual_outputs = {
    t.name:
    torch.empty(tuple(t.shape),
                dtype=trt_dtype_to_torch(t.dtype),
                device="cuda")
    for t in output_info
}
ok = encoder_session.run(visual_features, visual_outputs, stream.cuda_stream)
assert ok, "Runtime execution failed for encoder session"
stream.synchronize()

image_embeds = visual_outputs["encoder_output"]
image_embeds = image_embeds.reshape(55, 55,
                                    -1)[:h // 28, :w // 28].flatten(0, 1)

print(image_embeds.shape)  # torch.Size([1504, 5120])
print(image_embeds)

# # From modeling_mistral3.py.
input_ids = inputs["input_ids"].to(device="cuda")
# image_token_index = 10
# special_image_mask = (input_ids == image_token_index).unsqueeze(-1)  # torch.Size([1, 1545, 1])
# special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)   # torch.Size([1, 1545, 5120])
# image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)   # torch.Size([1504, 5120])
# inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)    # torch.Size([1, 1545, 5120])

import numpy as np

vocab_size = 131072

# Replace all image tokens with a unique token_id > text_vacab_size. This shall be used to lookup the prompt table.
image_token_index = 10
replacer = vocab_size
for i in range(len(input_ids[0])):
    if input_ids[0][i] == image_token_index:
        input_ids[0][i] = replacer
        print("replacer: ", replacer)
        replacer += 1

prompt_tasks = ",".join(np.arange(1, dtype=np.int32).astype(str))
prompt_table = image_embeds
prompt_table = prompt_table.view(1, -1, prompt_table.shape[-1])

llm_engine_dir = "/home/bbuddharaju/scratch/TensorRT-LLM/mistral_mm_eng/llm/"
model = ModelRunnerCpp.from_dir(
    llm_engine_dir,
    rank=0,
    debug_mode=False,
)
model_config = model.model_config
generated_ids = model.generate(input_ids,
                               input_position_ids=None,
                               mrope_params=None,
                               encoder_input_features=None,
                               sampling_config=None,
                               max_new_tokens=15,
                               end_id=2,
                               pad_id=2,
                               num_beams=1,
                               prompt_table=prompt_table,
                               prompt_tasks=prompt_tasks,
                               output_sequence_lengths=False,
                               return_dict=False,
                               mm_embedding_offloading=False)
generated_text = processor.batch_decode(generated_ids[0],
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
print("generated_text: ", generated_text)
