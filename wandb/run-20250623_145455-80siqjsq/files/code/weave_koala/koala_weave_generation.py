
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import weave
import wandb
import numpy as np
import asyncio
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
os.chdir(parent_dir) # Set the file to run FROM the parent directory of this script. 
sys.path.insert(0, parent_dir) # Add the parent directory at front of sys path so imports work. 

print('Curr working dir ', os.getcwd())

from weave_koala.KoalaClasses import KoalaModel, KoalaBinaryScorer, KoalaCategoryScorer
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
with open('./datasets/responses-375.json', 'r') as file:
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