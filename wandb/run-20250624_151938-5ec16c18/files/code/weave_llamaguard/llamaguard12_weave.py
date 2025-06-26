import json
import pandas as pd
import weave
import wandb
import numpy as np
import asyncio
import os, sys
from weave import Evaluation
import requests
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
os.chdir(parent_dir) # Set the file to run FROM the parent directory of this script. 
sys.path.insert(0, parent_dir) # Add the parent directory at front of sys path so imports work. 
load_dotenv('./login.env')
from weave_llamaguard.LLamaGuard12B import LGModel, LGCategoryScorer, LGBinaryScorer


print(requests.get(os.getenv('WANDB_HOST')))
# Could set these in config 
PROJECT_NAME = "dataset-severitylevels-104"
ENTITY = "liammurphy7657"

wandb.login(host=os.getenv('WANDB_HOST'), key=os.getenv('WANDB_API_KEY'))  # Login to Weave
weave.init(project_name=PROJECT_NAME)
wandb.init(project=PROJECT_NAME, entity=ENTITY)


# Load the dataset from the JSON file
with open(os.getenv("DATASET_PATH"), "r") as file:
    data = json.load(file)
# Run the evaluation using weave Evaluation.
model = LGModel()
cat_accuracy = LGCategoryScorer()
binary_accuracy = LGBinaryScorer()
evaluation = Evaluation(dataset=data, scorers=[cat_accuracy, binary_accuracy])

results = asyncio.run(
    evaluation.evaluate(model)
)  # Forwards results to website/database

model.make_confusion_matrix()
print(results)
