"""
Detoxify model
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import weave
from weave import Model, Scorer
import torch
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from typing import Optional

hf_model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name)

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]

"""Weave Model that is based off of the hugging face model"""


# Inherits from weave.Model
class DetoxifyModel(Model):  # Class just to process/predict input.
    # Inputs a prompt, outputs multiple probabilities.
    weightings: list
    severity_labels: list
    base_dataset: list
    sorted_centroids: Optional[list] = None

    def __init__(self, weightings, severity_labels, y_true):
        super().__init__(
            weightings=weightings, severity_labels=severity_labels, base_dataset=y_true
        )
    @weave.op()
    def predict(self, question: str) -> dict:  #: str just clarifies it is a stirng
        inputs = tokenizer(question, return_tensors="pt")  # 2 arguments
        output = hf_model(**inputs)
        predictions = torch.sigmoid(
            output.logits
        )  # Converts logits to probabilities for multiple labels
        predictions = predictions.detach().numpy().flatten()
        weighted = predictions * np.array(self.weightings)
        predicted_label = self.severity_labels[
            self.classify_severity(self.sorted_centroids, weighted)
        ]
        # rescaled = 1/ (1 + np.exp(-weighted))
        return weighted

    @weave.op()
    def classify_severity(self, sorted_centroids, pred_point):  # Outputs index
        pred_point = np.array(pred_point)

        # Originally using euclidian distance. Cosine approach is at the bottom
        distances = []
        for cen in sorted_centroids:
            distances.append(distance.euclidean(pred_point, cen))
        return np.argmin(distances)

    def create_clusters(self, y_true):
        """
        Then, create the initial clusters with KMeans. Get the cluster centroids
        Returns the KMeans objkect and sets the sorted centroids in the instance vars
        """
        kmeans = KMeans(n_clusters=4, random_state=5).fit(y_true)
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids, axis=0)
        self.sorted_centroids = np.take_along_axis(centroids, sorted_indices, axis=0)
        return kmeans

# class SeverityLevelScorer(Scorer):
#     @weave.op(name="severity_scorer")
#     def score(self, expected_severity_level: str, output: dict) -> dict:
#         return {
#             "expected_severity_level": expected_severity_level,
#         }
