import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import numpy as np
import json

login("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")

model_id = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

class LLamaGuardModel():
    def __init__(self):
        pass
    def predict(chat):
        sys_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        print(sys_prompt)
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt")
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0) # Takes argmax output and feeds it into the model until <eot>
        prompt_lense = input_ids.shape[-1] # Size of sequence 
        return tokenizer.decode(output[0][prompt_lense:], skip_special_tokens = True).split() #extracts from size of input sequence to <eot>, which is our answer. 
# returns tokens 208 -> 214. 



# with open("./afg-moderation/responses_severity.json", "r") as file:  # Load our data
#     data = json.load(file)
    
# for entry in data:
#     prompt = entry['question']
#     print(prompt)
#     chat = [{"role": "user", "content": prompt}]
#     prediction = predict(chat)
#     print(f"Prediction: {prediction[-1]}")