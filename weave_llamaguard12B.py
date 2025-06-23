import json
import pandas as pd
import weave
import wandb
import numpy as np
import asyncio
import os
from weave_models.LLamaGuard12B import LGModel, LGBinaryScorer, LGCategoryScorer
from weave import Evaluation
import requests

response = requests.get("https://weave.assessmentforgood.cloud/health")
print(f"RESPONSE {response}")
os.environ.pop("WEAVE_DEBUG", None)
os.environ.pop("WANDB_API_KEY", None)
os.environ.pop("WANDB_BASE_URL", None)
os.environ.pop("WANDB_SERVER_URL", None)
os.environ["WEAVE_DEBUG"] = "2"
PROJECT_NAME = "llamaguard_classification"
ENTITY = "liammurphy7657"

wandb.login(host="https://weave.assessmentforgood.cloud/", key="local-675e74f10414fc93a4349cccb95e19ed62aa2d58")  # Login to Weave
weave.init(project_name=PROJECT_NAME)
wandb.init(project=PROJECT_NAME, entity=ENTITY)


# Load the dataset from the JSON file
with open("afg-moderation/datasets/responses-378.json", "r") as file:
    data = json.load(file)
    # data = data[:100]

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
