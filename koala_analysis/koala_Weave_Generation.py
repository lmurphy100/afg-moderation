
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import pandas as pd
import torch
import weave
import wandb
import numpy as np
import asyncio
import os
from KoalaClasses import KoalaModel, AccuracyScorer
from weave import Evaluation, Model
import requests


response = requests.get('https://weave.assessmentforgood.cloud/health')
print(f"RESPONSE {response}")
os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
wandb.login(key="local-7a2558c504e83326890cab75e17017b23327bf77")  # Get from wandb.ai/settings

os.environ["WEAVE_DEBUG"] = "2"
os.environ["WANDB_DEBUG"] = "true"

PROJECT_NAME = "afg-safety-module"
ENTITY = 'liamm76'
wandb.init(project=PROJECT_NAME, entity=ENTITY)
weave.init(PROJECT_NAME, )

with open('afg-moderation/responses.json', 'r') as file:
    data = json.load(file)  # convert to dict

model = KoalaModel()   
accuracy = AccuracyScorer()
evaluation = Evaluation(dataset=data, 
                        scorers=[accuracy]
) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
print(results)
print(len(data))