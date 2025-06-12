import IPython

print(IPython.__file__)
import matplotlib.pyplot as plt
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from scipy.spatial import distance
import wandb
import weave
import asyncio
from weave import Evaluation, Model, Scorer
import mplcursors
from mplcursors import cursor
from weave_models.DetoxifyClasses import DetoxifyModel
import os

# os.environ["WEAVE_DEBUG"] = "1"  # Show detailed serialization logs

validation_data = "./afg-moderation/datasets/detoxify_validation.csv"
response_data = "./afg-moderation/datasets/responses-375.json"
labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
severity_labels = ["N/A", "Informational", "Questionable", "Urgent"]
y_true = (
    []  # Exisiting toxicity scores expected from the VALIDATION data. We can weight these.
)
weightings = [1, 1, 1, 1, 1, 1]

with open(response_data, "r") as file:  # Load our data
    data = json.load(file)

val_data = pd.DataFrame(pd.read_csv(validation_data)).to_dict()
y_true = []
for idx in range(len(val_data)):
    values = []
    for label in labels:
        values.append(val_data[label][idx])
    weighted_values = values * np.array(weightings)
    y_true.append(weighted_values)
# Get true scores in the validation set, for clustering

detoxify_model = DetoxifyModel(
    weightings=weightings,
    severity_labels=severity_labels,
    y_true=y_true
)
kmeans = detoxify_model.create_clusters(y_true)

results_data = []

for entry in data:
    result = detoxify_model.predict(entry["response"])
    results_data.append({
        'predicted_toxicity_scores': result
    })
results_df = pd.DataFrame(results_data)  # Convert results to a DataFrame
print(results_df)

"""
PCA can
-
- Tell us which features contribute to ethier axis
- Tell us how accurate the 2D Graph is. In this case, reducing to 2 dimensions preserves 90% of the data variation.
"""

pca = PCA(n_components=2)
X_pca = pca.fit_transform(y_true)
# PC1, PC2 preserve 90% of the variance. PC1, PC2, PC3 preserve 95% of the variance.
variances_axis = pca.explained_variance_ratio_
loadings = pca.components_
print(f" Variances: {variances_axis}")
print(f"Loadings Per Feature for Component: {loadings}")

y_pred = results_df[
    "predicted_toxicity_scores"
].tolist() 
y_pred_pca = pca.transform(y_pred)
y_pred_pca = np.array(y_pred_pca)  #

PC1 = X_pca[:, 0]  # 1 dimensional, column one
PC2 = X_pca[:, 1]
scalePCA = 1

fig, ax1 = plt.subplots(
    figsize=(10, 8)
)  # ax1 - original data,   #ax2 - our data plotted on it
scatter = ax1.scatter(
    x=PC1,
    y=PC2,
    c=kmeans.labels_,
    marker=".",
)

for i, feature in enumerate(labels):
    ax1.arrow(
        0, 0, dx=loadings[0, i] * scalePCA, dy=loadings[1, i] * scalePCA, color="red"
    )  # Plot arrows
    ax1.text(
        x=loadings[0, i] * (scalePCA),
        y=loadings[1, i] * (scalePCA),
        s=feature,
        color="black",
    )

ax1.set_title("PCA Analysis with 3 Clusters")
ax1.set_xlabel(f"PCA 1 - {(variances_axis[0]*100):.2f}%")
ax1.set_ylabel(f"PCA 2 - {(variances_axis[1]*100):.2f}%")
legend = ax1.legend(*scatter.legend_elements(), title="Clusters", loc="lower left")

# Plot predictions as red dots
predScatter = ax1.scatter(
    y_pred_pca[:, 0],
    y_pred_pca[:, 1],
    color="red",
    marker="X",
    s=30,
    label="Predictions",
)

cursor = mplcursors.cursor(predScatter, hover=True)


@cursor.connect("add")
def on_hover(sel):
    index = sel.index
    x, y = sel.target
    report = (
        f"Coordinates: ({x:.2f}, {y:.2f}) - Index: {index}\n"
        f"Prompt: {results_df.iloc[index, 0]}\n\n"
        f"Predicted Severity Level: {(results_df.iloc[index, 2])}\n"
        f"Expected Content Label: {results_df.iloc[index, 6]}\n\n"
        f"Predicted Toxicity Score: {results_df.iloc[index, 7]:.4f}\n"
        f"Predicted Severe Toxicity Score: {results_df.iloc[index, 8]:.4f}\n"
        f"Predicted Obscene Score: {results_df.iloc[index, 9]:.4f}\n"
        f"Predicted Threat Score: {results_df.iloc[index, 10]:.4f}\n"
        f"Predicted Insult Score: {results_df.iloc[index, 11]:.4f}\n"
        f"Predicted Identity Attack Score: {results_df.iloc[index, 12]:.4f}\n"
    )
    print(f"{report}")
    sel.annotation.set(
        text=report,
        bbox=dict(fc="white", alpha=0.8),
        wrap="true",
    )


plt.tight_layout()
plt.show()
plt.savefig('pac_analysis.png')


   # - Cosine similarity approach
    # for cen in sorted_centroids:
    #     cen = np.array(cen)
    #     similar_direction = (np.dot(pred_point, cen)/(np.linalg.norm(pred_point) + np.linalg.norm(cen)))
    #     distances.append(similar_direction)
