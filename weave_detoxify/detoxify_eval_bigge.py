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
from mplcursors import cursor
import os, sys
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

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

response_data = (
    "C:/Users/LiamMurphy(DMC)/Documents/Weave/eval/datasets/responses-378.json"
)
test_data = (
    "C:/Users/LiamMurphy(DMC)/Documents/Weave/eval/datasets/sensitive-ex-reports.json"
)
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

# all_categories = ['safe', 'report of abuse', 'self harm']
filter_categories = ["safe", "self harm"]
severity_labels = ["n/a", "informational", "questionable", "urgent"]


# def make_confusion_matrix():
#     encoder = LabelEncoder().fit(severity_labels)

#     y_pred_encoded = encoder.transform(y_pred_label_test)
#     y_true_encoded = encoder.transform(y_true_label_test)

#     report_dict = classification_report(
#         y_pred=y_pred_encoded,
#         y_true=y_true_encoded,
#         target_names=severity_labels
#     )
#     print(report_dict)

#     cm = confusion_matrix(y_true_encoded, y_pred_encoded)
#     sns.heatmap(
#         cm,
#         annot=True,
#         fmt="d",
#         cmap="Blues",
#         cbar=False,
#         linewidths=0.5,
#         linecolor="lightgray",
#         xticklabels=severity_labels,
#         yticklabels=severity_labels,
#         annot_kws={"size": 12},
#     )
#     plt.xlabel("Predicted Severity")
#     plt.ylabel("Actual Severity")
#     plt.title(" VS ".join(filter_categories))


def predict(response: str) -> dict:
    inputs = tokenizer(response, return_tensors="pt")  # 2 arguments
    output = hf_model(**inputs)
    softmax_prob = torch.softmax(output.logits, dim=1)
    sigmoid_prob = torch.sigmoid(output.logits)
    prob = softmax_prob.detach().numpy().flatten()
    sigmoid_prob = sigmoid_prob.detach().numpy().flatten()
    return prob


with open(response_data, "r") as file:  # Load our data we used to generate cluster
    data = json.load(file)

df = pd.DataFrame(data)
data = df[(df["category"].isin(filter_categories))].to_dict(orient="records")


for entry in data:
    prob = predict(entry["response"])
    y_pred.append(prob)
    prompts.append(entry["response"])
    category.append(entry["category"])

print(len(data))

k = len(severity_labels)
n = 3

# Creating our clusters made with the 378-response data to base our new dataset on
kmeans = KMeans(n_clusters=k, random_state=42).fit(y_pred)
centroids = kmeans.cluster_centers_

max_scores = np.max(centroids, axis=1)
severity_order = np.argsort(max_scores)
print(severity_order)
cluster_to_severity = {
    severity_order[0]: severity_labels[0],
    severity_order[1]: severity_labels[1],
    severity_order[2]: severity_labels[2],
    severity_order[3]: severity_labels[3],
}
label_to_index = {label: i for i, label in enumerate(severity_labels)}
print(label_to_index)

cluster_labels = kmeans.labels_
# Check how many 'safe' categories are in each cluster.
df_clusters = pd.DataFrame(
    {"prompt": prompts, "category": category, "cluster": cluster_labels}
)
df_clusters["cluster"] = df_clusters["cluster"].map(cluster_to_severity)
count_per_cluster = (
    df_clusters[df_clusters["category"] == "safe"].groupby("cluster").size()
)

counts = pd.crosstab(df_clusters["cluster"], df_clusters["category"])
print(counts)


knn = KNeighborsClassifier(n_neighbors=n)
mapped_cluster_labels = pd.Series(cluster_labels).map(cluster_to_severity)
mapped_cluster_ids = mapped_cluster_labels.map(label_to_index)
knn.fit(y_pred, mapped_cluster_ids)

print(mapped_cluster_labels)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(y_pred)

PC1 = X_pca[:, 0]
PC2 = X_pca[:, 1]
scalePCA = 1

fig, ax1 = plt.subplots(figsize=(10, 8))
# plot per cluster.
indexes = []
cluster_scatters = []
for i in range(k):
    idx = np.where(i == cluster_labels)[0]
    scatter = ax1.scatter(PC1[idx], PC2[idx], marker=".", label=cluster_to_severity[i])
    indexes.append(idx)
    cluster_scatters.append(scatter)
    
cursor = mplcursors.cursor(cluster_scatters, hover=True)


@cursor.connect("add")
def on_hover(sel):
    cluster_idx = cluster_scatters.index(sel.artist)
    true_idx = indexes[cluster_idx][sel.index]
    index = true_idx
    x, y = sel.target
    report = (
        f"Coordinates: ({x:.2f}, {y:.2f}) - Index: {index}\n"
        f"Prompt: {prompts[index]}\n\n"
        f"Expected Category: {category[index]}\n"
        f"Predicted Severity Score: {y_pred[index]}\n\n"
        # f"Predicted Severity Label: {severity_labels[index]}"
    )
    print(f"{report}")
    sel.annotation.set(
        text=report,
        bbox=dict(fc="white", alpha=0.8),
        wrap="true",
    )


ax1.set_title("Severity Levels: Report of Abuse VS Safe")
ax1.set_xlabel(f"PCA 1")
ax1.set_ylabel(f"PCA 2")
plt.legend()
plt.show()
