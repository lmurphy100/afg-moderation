
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import pandas as pd
import weave
import wandb
import numpy as np
import asyncio
import os
from weave_models.KoalaClasses import KoalaModel, KoalaBinaryScorer, KoalaCategoryScorer
from weave import Evaluation
import requests

response = requests.get('https://weave.assessmentforgood.cloud/health')
print(f"RESPONSE {response}")
os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
os.environ["WEAVE_DEBUG"] = "2"
os.environ["WEAVE_STORAGE_DIR"] = "C://weave_traces"  
PROJECT_NAME = "koala_classification"
ENTITY = 'liammurphy7657'
weave.init("koala_classification") 

# Login to Weave AFG Cloud Instance
wandb.login(host="https://weave.assessmentforgood.cloud/", key="local-675e74f10414fc93a4349cccb95e19ed62aa2d58")  # Login to Weave


wandb.init(project=PROJECT_NAME, entity=ENTITY)

# Load the dataset from the JSON file
with open('afg-moderation/datasets/responses-375.json', 'r') as file:
    data = json.load(file)  

# Run the evaluation using weave Evaluation. 
model = KoalaModel()   
cat_accuracy = KoalaCategoryScorer()
binary_accuracy = KoalaBinaryScorer()
evaluation = Evaluation(dataset=data, 
                        scorers=[cat_accuracy, binary_accuracy],
) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
print(results)