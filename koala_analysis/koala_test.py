
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import pandas as pd
import duckdb
import torch
import weave
import numpy as np
import asyncio

from weave import Evaluation, Model
weave.init('koala_classification')

koala_labels = ["H", "H2", "HR", "OK", "S", "S3","SH","V","V2"]
content_labels = ["safe", "sexual content", "hate speech", "violence", "reports of abuse"]

'''
Conversion:
Because Koala's outputted labels don't directly match with our content labels. I had
to fit which koala labels we can use. 

'''
conversion = {
    "H":"hate speech",
    "H2":"hate speech",
    "HR":"reports of abuse",
    "OK":"safe",
    "S":"sexual content",
    "S3":"reports of abuse",
    "SH":"N/A",
    "V":"violence",
    "V2":"violence",
}
'''
Model for the predict Function. 
'''
class KoalaModel(Model):  # Class just to process/predict input. 
    # Inputs a Prompt to output a label.
    @weave.op()
    def predict(self, question: str) -> dict:   #: str just clarifies it is a stirng
      inputs = tokenizer(question, return_tensors="pt") # 2 arguments
      output = hf_model(**inputs) # there are 2 arguments
      logits = output.logits
      probabilities = logits.softmax(dim=-1).squeeze()  # Normalizes the logits between 0-1 and flattens them
      probabilities = probabilities.detach().numpy() # Convert from tensor to array
      output = koala_labels[np.argmax(probabilities)]
      return {'generated_label':output}
        # # here's where you would add your LLM call and return the output

@weave.op()
def accuracy(expected: str, output: dict) -> dict:   # Put the return dict value in output.   Takes in 1 expected
  """
  expected: correct label the other dataset expects
  output: dict format if the model is correct
  """
  label = conversion[output['generated_label']] # Converts to one of the 5 content koala_labels

  return {'match': label == expected,
          'mapped_label':label,
          'predicted_label:':output['generated_label'],
          'true_label':expected,
          }   # match: True




with open('../responses.json', 'r') as file:
    data = json.load(file)  # convert to dict

model = KoalaModel()   
evaluation = Evaluation(dataset=data, 
                        scorers=[accuracy],
                        preprocess_model_input=lambda row: 
                            {'question': row['question'],
                             'expected': row['expected']} 
) 

results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database
print(results)
print(len(data))