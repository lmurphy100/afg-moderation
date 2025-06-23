import json
import pandas as pd
import weave
import wandb
import numpy as np
import asyncio
import os
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-Guard-4-12B"
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map='cuda'
)
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # doing device = 'cuda:0' is not a torch device type
print(f"Using device: {device}")


def predict(prompt):
    chat = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=0)
        response = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )[0].split()
    return response


with open("afg-moderation/datasets/responses-378.json", "r") as file:
    data = json.load(file)
    data = data[:2]

for entry in data:
    prompt = entry["response"]
    print(prompt)
    output = predict(prompt)
    print(f"\n Predicted: {output}")


# Run the evaluation using weave Evaluation.
results = []
