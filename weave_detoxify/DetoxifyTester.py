import matplotlib.pyplot as plt
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import mplcursors
import os, sys
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from weave import Model, Scorer
import weave, wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(parent_dir)  # Set the file to run FROM the parent directory of this script.
sys.path.insert(
    0, parent_dir
)  # Add the parent directory at front of sys path so imports work.
print("Current dir ", os.getcwd())
load_dotenv("./login.env")

hf_model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)

filter_categories = "report of abuse"

severity_labels = ["n/a", "informational", "questionable", "urgent"]
label_to_index = {label: i for i, label in enumerate(severity_labels)}

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]

categories = []
pred_scores = []  # Toxicity Scores on test data
pred_severity_labels = []  # Cluster labels
true_severity_labels = []  # True cluster labels
prompts = []


class DetoxifyModel(Model):
    @weave.op()
    def predict(self, response: str) -> dict:
        inputs = tokenizer(response, return_tensors="pt")  # 2 arguments
        output = hf_model(**inputs)
        softmax_prob = torch.softmax(output.logits, dim=1)
        prob = softmax_prob.detach().numpy().flatten().tolist()
        return {'pred_score': prob}

    def make_confusion_matrix(self):
        encoder = LabelEncoder().fit(severity_labels)

        y_pred_encoded = encoder.transform(pred_severity_labels)
        y_true_encoded = encoder.transform(true_severity_labels)
        report_dict = classification_report(
            y_pred=y_pred_encoded, y_true=y_true_encoded, target_names=severity_labels
        )
        print(report_dict)
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="lightgray",
            xticklabels=severity_labels,
            yticklabels=severity_labels,
            annot_kws={"size": 12},
        )
        plt.xlabel("Predicted Severity")
        plt.ylabel("Actual Severity")
        plt.title(" VS ".join(filter_categories))
        plt.show()
    def get_results(self):
        return pred_scores, pred_severity_labels, true_severity_labels, prompts, categories

class DetoxifyScorer(Scorer):
    mapped_severity: dict
    cluster_labels: list
    cluster_points: list
    @weave.op()
    def score(self, severity_level: str, category: str, response: str, output: dict) -> dict:
        prediction = output['pred_score']
        pred_severity_lab = severity_labels[self.classify_severity(prediction)]
        
        prompts.append(response)
        categories.append(category)
        pred_scores.append(prediction)
        pred_severity_labels.append(pred_severity_lab)
        true_severity_labels.append(severity_level)
        
        return {
            'pred_severity':str(pred_severity_lab),
            'true_severity':str(severity_level),
            'match_severity':str(pred_severity_lab) == str(severity_level)
        }
    def classify_severity(self, point):
        n = 3
        knn = KNeighborsClassifier(n_neighbors=n)
        mapped_cluster_labels = pd.Series(self.cluster_labels).map(self.mapped_severity)
        mapped_cluster_ids = mapped_cluster_labels.map(label_to_index)
        knn.fit(self.cluster_points, mapped_cluster_ids)
        
        return int(knn.predict([point]))
    