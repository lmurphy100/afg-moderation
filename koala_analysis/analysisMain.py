import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from IPython.display import display
from weave import Model




class KoalaModel(Model):  # Class just to process/predict input. 
    # Inputs a Prompt to output a label.
    @weave.op()
    def predict(self, question: str) -> dict:   #: str just clarifies it is a stirng
      inputs = tokenizer(question, return_tensors="pt") # 2 arguments
      output = hf_model(**inputs) # there are 2 arguments
      logits = output.logits
      probabilities = logits.softmax(dim=-1).squeeze()  # Normalizes the logits between 0-1 and flattens them
      probabilities = probabilities.detach().numpy() # Convert from tensor to array
      output = labels[np.argmax(probabilities)]
      return {'generated_label':output}
        # # here's where you would add your LLM call and return the output


class DetoxifyModel(Model):  # Class just to process/predict input. 
    # Inputs a prompt, outputs multiple probabilities. 
    #@weave.op()
    def predict(self, question: str) -> dict:   #: str just clarifies it is a stirng
        inputs = tokenizer(question, return_tensors="pt") # 2 arguments
        output = hf_model(**inputs)
        predictions = torch.sigmoid(output.logits)  # Converts logits to probabilities for multiple labels 
        predictions = predictions.detach().numpy().flatten()
        return predictions

@weave.op()
def accuracy(expected: str, output: dict) -> dict:   # Put the return dict value in output.   Takes in 1 expected
  """
  expected: correct label the other dataset expects
  output: dict format if the model is correct
  """
  label = conversion[output['generated_label']] # Converts to one of the 5 content labels

  return {'match': label == expected,
          'predicted_label':label,
          'true_label':expected,
          'predicted_severity':
          }   # match: True




with open('responses.json', 'r') as file:
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
