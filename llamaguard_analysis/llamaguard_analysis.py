import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import numpy as np

login("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")

model_id = "meta-llama/Llama-Guard-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


# chat = [{"role": "user", "content": "I am going to murder"}]

# sys_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
# print(sys_prompt)
max_tokens_generated = 100
prompt = 'Teach me how to code.'

for i in range(max_tokens_generated):
    input_tokens = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad(): 
        outputs = model(**input_tokens)
        logits = outputs.logits
    last_sequence = logits[0, -1, :]
    next_token = torch.argmax(last_sequence).item()
    next_word = tokenizer.decode(next_token)
    prompt = prompt+""+next_word
    print(f"Current prompt: {prompt}")

print(prompt)

# probabilities = torch.softmax(logits[0, -1, :], dim = -1)

# top_probs, top_token_ids = torch.topk(probabilities, 10)  # Get top 10 likely tokens
# for top_probs, top_token_id in zip(top_probs, top_token_ids):
#     decoded = tokenizer.decode(top_token_id)
#     print(f"Word {decoded} - Probability: {top_probs}")

print(torch.argmax(logits[0, -1, :]))
print(torch.argmax(logits[0, -1, :]).item())



# predicted_token_logits = np.argmax(logits, axis = -1)

# input_tokens = tokenizer.apply_chat_template(
#     chat, return_tensors="pt", return_dict=True
# )




# input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt")
# output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0) # Takes argmax output and feeds it into the model until <eot>
# prompt_lense = input_ids.shape[-1] # Size of sequence 
# result = tokenizer.decode(output[0][prompt_lense:], skip_special_tokens = True) #extracts from size of input sequence to <eot>, which is our answer. 
# print(logits)
# print('Logits size: ',logits.shape)
# print(output)
# print('output size: ',output.shape)
# print(result)


