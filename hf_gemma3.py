from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch
import numpy as np

# Seeding for deterministic sampling.
torch.manual_seed(42)
np.random.seed(42)

model_path = "/home/scratch.trt_llm_data/llm-models/gemma/gemma-3-1b-it/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = "Hello, how are you?"
input_tokens = tokenizer(input_text, return_tensors="pt")

model = Gemma3ForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).eval()

output_tokens = model.generate(input_tokens["input_ids"], max_length=100, do_sample=False, num_beams=1, top_p=None, top_k=None, temperature=None)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
