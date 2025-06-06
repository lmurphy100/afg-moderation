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

os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
wandb.login(key="local-7a2558c504e83326890cab75e17017b23327bf77")  # Get from wandb.ai/settings

response = requests.get('https://weave.assessmentforgood.cloud/health')
print(f"RESPONSE {response}")

PROJECT_NAME = "afg-safety-module"
ENTITY = 'liamm76'
wandb.init(project=PROJECT_NAME, entity=ENTITY)
weave.init(PROJECT_NAME, )



trace_url = 'https://weave.assessmentforgood.cloud/traces/table/create'
test_payload = {"test": "data"}
headers = {
    'Content-Type':"application/json",
}

response = requests.post(url=trace_url, json=test_payload, headers=headers)
print(f"Response status code: {response.status_code} - Message: {response.text}")

# table = pd.DataFrame({'test':[1, 2, 3]})
# weave.publish(obj = table, name='test table')
