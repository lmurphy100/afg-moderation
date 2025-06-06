
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import pandas as pd
import torch
import weave
import numpy as np
import asyncio
from KoalaClasses import KoalaModel, AccuracyScorer

from weave import Evaluation, Model
weave.init('koala_classification')
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