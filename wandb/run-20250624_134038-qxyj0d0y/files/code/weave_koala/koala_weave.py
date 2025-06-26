
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import weave
import wandb
import numpy as np
import asyncio
import os, sys
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
os.chdir(parent_dir) # Set the file to run FROM the parent directory of this script. 
sys.path.insert(0, parent_dir) # Add the parent directory at front of sys path so imports work. 
load_dotenv('./login.env')

from weave_koala.KoalaClasses import KoalaModel, KoalaBinaryScorer, KoalaCategoryScorer
from weave import Evaluation
import requests

response = requests.get(os.getenv('WANDB_HOST'))
print(f"RESPONSE {response}")
PROJECT_NAME = "dataset-severitylevels-104"
ENTITY = 'liammurphy7657'
weave.init(PROJECT_NAME) 

# Login to Weave AFG Cloud Instance
wandb.login(host=os.getenv('WANDB_HOST'), key=os.getenv('WANDB_API_KEY'))  # Login to Weave
wandb.init(project=PROJECT_NAME, entity=ENTITY)

# Load the dataset from the JSON file
with open(os.getenv("DATASET_PATH"), "r") as file:
    data = json.load(file)

# Run the evaluation using weave Evaluation. 
model = KoalaModel()   
cat_accuracy = KoalaCategoryScorer()
binary_accuracy = KoalaBinaryScorer()
evaluation = Evaluation(dataset=data, 
                        scorers=[cat_accuracy, binary_accuracy],
) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
model.make_confusion_matrix()
print(results)