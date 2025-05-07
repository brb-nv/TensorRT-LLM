import json
import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from tensorrt_llm.runtime.session import Session, TensorInfo
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 torch_dtype_to_trt, trt_dtype_to_torch)

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

url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
raw_image = Image.open(requests.get(url, stream=True).raw)

model_path = "/home/bbuddharaju/scratch/random/hf_models/Mistral-Small-3.1-24B-Instruct-2503/"
processor = AutoProcessor.from_pretrained(model_path)
inputs = processor(text="dummy", images=[raw_image], return_tensors="pt").to(
    str_dtype_to_torch(precision))

dtype = str_dtype_to_torch(precision)
d_min = torch.finfo(dtype).min
pixel_values = torch.full((1, 3, 1540, 1540), fill_value=0, dtype=dtype, device="cuda")
attention_mask = torch.full((1, 110, 110), fill_value=d_min, dtype=dtype, device="cuda")

_pixel_values = inputs["pixel_values"].to(device="cuda", dtype=dtype)
h, w = _pixel_values.shape[-2:]
pixel_values[..., :h, :w] = _pixel_values
attention_mask[..., :h // 14, :w // 14] = 0

visual_features = {"input": pixel_values, "attention_mask": attention_mask}
tensor_info = [
    TensorInfo("input", str_dtype_to_trt(precision), pixel_values.shape),
    TensorInfo("attention_mask", str_dtype_to_trt(precision),
               attention_mask.shape),
]
output_info = encoder_session.infer_shapes(tensor_info)
encoder_session.set_shapes(visual_features)
visual_outputs = {
    t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype),
                        device="cuda") for t in output_info
}
ok = encoder_session.run(visual_features, visual_outputs, stream.cuda_stream)
assert ok, "Runtime execution failed for encoder session"
stream.synchronize()

image_embeds = visual_outputs["encoder_output"]
image_embeds = image_embeds.reshape(55, 55, -1)[:h // 28, :w // 28].flatten(0, 1)

print(image_embeds.shape)
print(image_embeds)
