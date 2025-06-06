import IPython
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import weave
import asyncio
from weave import Evaluation
from mplcursors import cursor
from DetoxifyClasses import DetoxifyModel, SeverityLevelScorer
import sys

validation_data = "./afg-moderation/detoxify_analysis/detoxify_validation.csv"
response_data = "./afg-moderation/responses_severity.json"
labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
severity_labels = ["N/A", "Informational", "Questionable", "Urgent"]
weightings = [1, 1, 1, 1, 1, 1]
val_data = pd.DataFrame(pd.read_csv(validation_data)).to_dict()  # Pass this in


with open(response_data, "r") as file:  # Load our data
    data = json.load(file)


# Get true scores in the validation set, for clustering

weave.init("detoxify_severity_classification")
detoxify_model = DetoxifyModel(
    name="detoxify model",
    weightings=weightings,
    severity_labels=severity_labels,
    cluster_dataset=val_data,
)
severity_scorer = SeverityLevelScorer()
evaluation = Evaluation(
    dataset=data,
    scorers=[severity_scorer],
)
results = asyncio.run(
    evaluation.evaluate(detoxify_model)
)  # Forwards results to website/database

# print(results)
# results_data = []
# for entry in data:
#     result = severity_scorer.score(
#         expected_severity_level=entry["expected_severity_level"],
#         output=detoxify_model.predict(entry["question"]),
#
#     results_data.append(result)
# results_df = pd.DataFrame(results_data)  # Convert results to a DataFrame
# print(results_df)

"""
PCA can
- 
- Tell us which features contribute to ethier axis
- Tell us how accurate the 2D Graph is. In this case, reducing to 2 dimensions preserves 90% of the data variation. 
"""

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(y_true)
# # PC1, PC2 preserve 90% of the variance. PC1, PC2, PC3 preserve 95% of the variance.
# variances_axis = pca.explained_variance_ratio_
# loadings = pca.components_
# print(f" Variances: {variances_axis}")
# print(f"Loadings Per Feature for Component: {loadings}")

# y_pred = results_df[
#     "predicted_toxicity_scores"
# ].tolist()  # Convert predicted scores in df column to an array
# y_pred_pca = pca.transform(y_pred)
# y_pred_pca = np.array(y_pred_pca)  #

# PC1 = X_pca[:, 0]  # 1 dimensional, column one
# PC2 = X_pca[:, 1]
# scalePCA = 1

# fig, ax1 = plt.subplots(
#     figsize=(10, 8)
# )  # ax1 - original data,   #ax2 - our data plotted on it
# scatter = ax1.scatter(
#     x=PC1,
#     y=PC2,
#     c=kmeans.labels_,
#     marker=".",
# )

# for i, feature in enumerate(labels):
#     ax1.arrow(
#         0, 0, dx=loadings[0, i] * scalePCA, dy=loadings[1, i] * scalePCA, color="red"
#     )  # Plot arrows
#     ax1.text(
#         x=loadings[0, i] * (scalePCA),
#         y=loadings[1, i] * (scalePCA),
#         s=feature,
#         color="black",
#     )

# ax1.set_title("PCA Analysis with 3 Clusters")
# ax1.set_xlabel(f"PCA 1 - {(variances_axis[0]*100):.2f}%")
# ax1.set_ylabel(f"PCA 2 - {(variances_axis[1]*100):.2f}%")
# legend = ax1.legend(*scatter.legend_elements(), title="Clusters", loc="lower left")

# # Plot predictions as red dots
# predScatter = ax1.scatter(
#     y_pred_pca[:, 0],
#     y_pred_pca[:, 1],
#     color="red",
#     marker="X",
#     s=30,
#     label="Predictions",
# )

# cursor = mplcursors.cursor(predScatter, hover=True)


# @cursor.connect("add")
# def on_hover(sel):
#     index = sel.index
#     x, y = sel.target
#     report = (
#         f"Coordinates: ({x:.2f}, {y:.2f}) - Index: {index}\n"
#         f"Prompt: {results_df.iloc[index, 0]}\n\n"
#         f"Predicted Severity Level: {(results_df.iloc[index, 2])}\n"
#         f"Expected Content Label: {results_df.iloc[index, 6]}\n\n"
#         f"Predicted Toxicity Score: {results_df.iloc[index, 7]:.4f}\n"
#         f"Predicted Severe Toxicity Score: {results_df.iloc[index, 8]:.4f}\n"
#         f"Predicted Obscene Score: {results_df.iloc[index, 9]:.4f}\n"
#         f"Predicted Threat Score: {results_df.iloc[index, 10]:.4f}\n"
#         f"Predicted Insult Score: {results_df.iloc[index, 11]:.4f}\n"
#         f"Predicted Identity Attack Score: {results_df.iloc[index, 12]:.4f}\n"
#     )
#     print(f"{report}")
#     sel.annotation.set(
#         text=report,
#         bbox=dict(fc="white", alpha=0.8),
#         wrap="true",
#     )


# plt.tight_layout()
# plt.show()
# # plt.savefig('pac_analysis.png')


# #    # - Cosine similarity approach
# #     # for cen in sorted_centroids:
# #     #     cen = np.array(cen)
# #     #     similar_direction = (np.dot(pred_point, cen)/(np.linalg.norm(pred_point) + np.linalg.norm(cen)))
# #     #     distances.append(similar_direction)
