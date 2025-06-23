from transformers import AutoTokenizer, AutoModelForSequenceClassification
import weave
from weave import Model, Scorer
import torch
import json
import numpy as np
with open("afg-moderation/config.json", "r") as f:
    config = json.load(f)
koala2cat = config["koala2cat"]
koala2binary = config["koala2binary"]
cat2binary = config["cat2binary"]
y_pred = []

device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "KoalaAI/Text-Moderation").to(device)

koala_labels = koala_labels = ["H", "H2", "HR", "OK", "S", "S3", "SH", "V", "V2"]
content_labels = [
    "safe",
    "sexual content",
    "hate speech",
    "violence",
    "reports of abuse",
]

class KoalaModel(Model):  # Class just to process/predict input.

    # Inputs a Prompt to output a label.
    @weave.op(name='accuracy_scorer')
    def predict(self, response: str) -> dict:  #: str just clarifies it is a stirng
        inputs = tokenizer(response, return_tensors="pt").to(device)
        output = hf_model(**inputs)  # there are 2 arguments
        logits = output.logits
        probabilities = logits.softmax(    # Tensor
            dim=-1
        ).squeeze() 
        probabilities = probabilities.cpu().detach().numpy()  # Convert from tensor to array
        output = koala_labels[np.argmax(probabilities)]
        return {
            "pred": output}
        # # here's where you would add your LLM call and return the output
    def generate_confusion_matrix(y_true, type):
        pass


class KoalaCategoryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        label = koala2cat[
            output["pred"]
        ]  # Converts to one of the 5 content koala_labels
        return {
            "mapped_category": label,
            "match_category": label == category,
        }  

class KoalaBinaryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        number = koala2binary[   # 1 if safe, 0 if not safe
            output["pred"]
        ]  
        expected_num = cat2binary[category]  # 1 if safe, 0 if not safe
        print(f"Predicted: {number}, Expected: {expected_num}")
        return {
            "mapped_binary": number,
            "match_binary": number == expected_num,
        } 