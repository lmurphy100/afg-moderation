
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
from weave_models.KoalaClasses import KoalaModel, AccuracyScorer
from weave import Evaluation, Model
from wandb import login
import requests


response = requests.get('https://weave.assessmentforgood.cloud/health')
print(f"RESPONSE {response}")
os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
os.environ["WEAVE_DEBUG"] = "2"

# Login to Weave Local Instance
#wandb.login(host="http://localhost:8080/", key="local-d04c4c8c56b8805773ce19e673f48bed5adbbea1")

# Login to Weave AFG Cloud Instance
#wandb.login(host="https://weave.assessmentforgood.cloud/", key="local-7a2558c504e83326890cab75e17017b23327bf77")  # Login to Weave

# Login to Weave Web Instance 
wandb.login(host="https://api.wandb.ai", key="ea6a9a44c1a47d814e00755570e7314a1304bf87")
PROJECT_NAME = "koala_classification"
ENTITY = 'liamm76'
#wandb.init(project=PROJECT_NAME, entity=ENTITY)
weave.init(PROJECT_NAME)

# success = login(key = 'ea6a9a44c1a47d814e00755570e7314a1304bf87') # Personal weave account API key
with open('afg-moderation/datasets/responses-375.json', 'r') as file:
    data = json.load(file)  # convert to dict

model = KoalaModel()   
accuracy = AccuracyScorer()
evaluation = Evaluation(dataset=data, 
                        scorers=[accuracy]
) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
print(results)
print(len(data))


'''
curl -X POST http://localhost:8080/traces -H "Content-Type: application/json" -d '{"test":"data"}

curl -X POST -H "Content-Type: application/json" -d '{"test":"data"}' http://localhost:8080/traces
curl.exe -X POST -H "Content-Type: application/json" -d '{"test":"data"}' http://localhost:8080/api/traces


curl -Method Get -Uri http://localhost:8080/traces/table/create


'''