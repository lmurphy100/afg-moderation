import matplotlib.pyplot as plt
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
severity_labels = severity_labels.sort()  # alphabetic

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
category = []
y_pred = []  # Toxicity Scores on larger data
prompts = []

category_test = []
y_pred_test = []  # Toxicity Scores on test data
y_pred_id_test = []  # Cluster ids
y_pred_label_test = []  # Cluster labels
y_true_label_test = []  # True cluster labels
prompts_test = []


class DetoxifyModel():
    def predict(self, response: str) -> dict:
        inputs = tokenizer(response, return_tensors="pt")  # 2 arguments
        output = hf_model(**inputs)
        softmax_prob = torch.softmax(output.logits, dim=1)
        prob = softmax_prob.detach().numpy().flatten()
        return {'pred_score': prob}

    def make_confusion_matrix():
        encoder = LabelEncoder().fit(severity_labels)

        y_pred_encoded = encoder.transform(y_pred_label_test)
        y_true_encoded = encoder.transform(y_true_label_test)

        report_dict = classification_report(
            y_pred=y_pred_encoded, y_true=y_true_encoded, target_names=severity_labels
        )
        print(report_dict)

# class DetoxifyScorer():
#     def score(self, category: str, output: dict) -> dict:
        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
        # cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        # sns.heatmap(
        #     cm,
        #     annot=True,
        #     fmt="d",
        #     cmap="Blues",
        #     cbar=False,
        #     linewidths=0.5,
        #     linecolor="lightgray",
        #     xticklabels=severity_labels,
        #     yticklabels=severity_labels,
        #     annot_kws={"size": 12},
        # )
        # plt.xlabel("Predicted Severity")
        # plt.ylabel("Actual Severity")
        # plt.title(" VS ".join(filter_categories))