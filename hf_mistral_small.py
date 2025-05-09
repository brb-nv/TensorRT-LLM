import requests
from PIL import Image
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

model_path = "/home/bbuddharaju/scratch/random/hf_models/Mistral-Small-3.1-24B-Instruct-2503"
model = Mistral3ForConditionalGeneration.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
prompt = "<s>[INST][IMG]What is the image?[/INST]"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print("[hf_mistral_small] raw image size: ", image.size)
inputs = processor(images=image, text=prompt, return_tensors="pt")
print("[hf_mistral_small] inputs: ", inputs)
generate_ids = model.generate(**inputs, max_new_tokens=15)
output_text = processor.batch_decode(generate_ids,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
print(output_text)
