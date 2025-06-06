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
    # name: str
    # weightings: list
    # severity_labels: list
    # base_dataset: dict
    # sorted_centroids: Optional[list] = None
    @weave.op()
    def __init__(self, name, weightings, severity_labels, cluster_dataset):
        self.name = name,
        self.weightings=weightings,
        self.severity_labels=severity_labels,
        self.base_dataset = cluster_dataset
        self.sorted_centroids = self.create_clusters(cluster_dataset)
        super().__init__()

    # @weave.op()
    # def __weave_model_serialize__(self):   # overrides default weave serialize. Which includes the whole class. 
    #     return None
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
        return {
            "predicted_severity_level": predicted_label,
            "predicted_toxicity_scores":weighted
        }

    @weave.op()
    def classify_severity(self, sorted_centroids, pred_point):  # Outputs index
        pred_point = np.array(pred_point)

        # Originally using euclidian distance. Cosine approach is at the bottom
        distances = []
        for cen in sorted_centroids:
            distances.append(distance.euclidean(pred_point, cen))
        return np.argmin(distances)
    @weave.op()
    def create_clusters(self, dataset):
        """
        First, get the scores from the inputted dataset (Currently from the validation set)
        Then, create the initial clusters with KMeans. Get the cluster centroids
        """
        y_true = []
        for idx in range(len(dataset)):
            values = []
            for label in labels:
                values.append(dataset[label][idx])
            values = values*np.array(self.weightings)
            y_true.append(values)
            
        y_true = np.array(y_true)
        
        kmeans = KMeans(n_clusters=4, random_state=5).fit(y_true)
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids, axis=0)
        return np.take_along_axis(centroids, sorted_indices, axis=0)

        


class SeverityLevelScorer(Scorer):
    @weave.op(name="severity_scorer")
    def score(self, expected_severity_level: str, output: dict) -> dict:
        return {
            "expected_severity_level": expected_severity_level,
        }
