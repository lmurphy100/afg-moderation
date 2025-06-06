from transformers import AutoTokenizer, AutoModelForSequenceClassification
import weave
from weave import Model, Scorer
import torch
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained(
    "KoalaAI/Text-Moderation"
)  # Actual model
koala_labels = koala_labels = ["H", "H2", "HR", "OK", "S", "S3", "SH", "V", "V2"]
content_labels = [
    "safe",
    "sexual content",
    "hate speech",
    "violence",
    "reports of abuse",
]
'''
Conversion:
Because Koala's outputted labels don't directly match with our content labels. I had
to fit which koala labels we can use. 

'''
mapping = {
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

class KoalaModel(Model):  # Class just to process/predict input.

    # Inputs a Prompt to output a label.
    @weave.op(name='accuracy_scorer')
    def predict(self, question: str) -> dict:  #: str just clarifies it is a stirng
        inputs = tokenizer(question, return_tensors="pt")  # 2 arguments
        output = hf_model(**inputs)  # there are 2 arguments
        logits = output.logits
        probabilities = logits.softmax(
            dim=-1
        ).squeeze()  # Normalizes the logits between 0-1 and flattens them
        probabilities = probabilities.detach().numpy()  # Convert from tensor to array
        output = koala_labels[np.argmax(probabilities)]
        return {
            "student response:":question,
            "predicted_content_label": output}
        # # here's where you would add your LLM call and return the output


class AccuracyScorer(Scorer):
    @weave.op()
    def score(self, expected: str, output: dict) -> dict:
        """ """
        label = mapping[
            output["predicted_content_label"]
        ]  # Converts to one of the 5 content koala_labels
        return {
            **output, 
            "mapped_content_label": label,
            "expected_content_label": expected,
        }  # match: True