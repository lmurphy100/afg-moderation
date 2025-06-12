import json
import pandas as pd
import weave
import wandb
import numpy as np
import asyncio
import os
from weave_models.LLamaGuardClasses import LGModel, LGCategoryScorer
from weave import Evaluation
import requests

response = requests.get('https://weave.assessmentforgood.cloud/health')
print(f"RESPONSE {response}")
os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
os.environ["WEAVE_DEBUG"] = "2"
os.environ["WEAVE_STORAGE_DIR"] = "C://weave_traces"  
PROJECT_NAME = "llamaguard_classification2"
ENTITY = 'liammurphy7657'

weave.init(project_name = PROJECT_NAME)
# Login to Weave AFG Cloud Instance
wandb.login(host="https://weave.assessmentforgood.cloud/", key="local-675e74f10414fc93a4349cccb95e19ed62aa2d58")  # Login to Weave
wandb.init(project=PROJECT_NAME, entity=ENTITY)



# Load the dataset from the JSON file
with open('afg-moderation/datasets/responses.json', 'r') as file:
    data = json.load(file)  

# Run the evaluation using weave Evaluation. 
model = LGModel()   
cat_accuracy = LGCategoryScorer()
evaluation = Evaluation(dataset=data, scorers=[cat_accuracy]) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
print(results)
