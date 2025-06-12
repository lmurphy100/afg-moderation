import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import numpy as np
import json
import weave
from weave import Model, Scorer


login("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")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

model_id = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='cuda')
print(model.device)

for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

class LGModel(Model):
    @weave.op(name="llama_guard_predict")
    def predict(self, response: str) -> dict:
    
        chat = [{"role": "user", "content": response}]

        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to('cuda:0') # Makes sure input ids on gpu
        output = model.generate(
            input_ids=input_ids, max_new_tokens=100, pad_token_id=0
        )  # Takes argmax output and feeds it into the model until <eot>
        prompt_lense = input_ids.shape[-1]  # Size of sequence
        result = tokenizer.decode(
            output[0][prompt_lense:], skip_special_tokens=True
        ).split()  # extracts from size of input sequence to <eot>, which is our answer. puts into array
        return {"llama_pred": result}

class LGCategoryScorer(Scorer):
    @weave.op(name="llama_guard_category_score")
    def score(self, category: str, output: dict) -> dict:
        return {
            "match_category": output["llama_pred"][0] == category,
        }
    # Returns safe or unsafe