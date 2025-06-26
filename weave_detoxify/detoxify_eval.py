import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import mplcursors
from mplcursors import cursor
import os, sys
from dotenv import load_dotenv
from weave_detoxify.DetoxifyTester import DetoxifyModel

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(parent_dir)  # Set the file to run FROM the parent directory of this script.
sys.path.insert(
    0, parent_dir
)  # Add the parent directory at front of sys path so imports work.
print("Current dir ", os.getcwd())
load_dotenv("./login.env")

response_data = (
    "C:/Users/LiamMurphy(DMC)/Documents/Weave/eval/datasets/responses-378.json"
)
test_data = (
    "C:/Users/LiamMurphy(DMC)/Documents/Weave/eval/datasets/sensitive-ex-reports.json"
)
severity_labels = ['informational', 'n/a', 'questionable', 'urgent']
categories = ['report of abuse']

# Load and filter out cluster and test datasets
with open(response_data, "r") as file:  # Load our data we used to generate cluster
    data = json.load(file)
with open(test_data, "r") as file:  # Load our testing data
    test_data = json.load(file)

df = pd.DataFrame(data)
data = df[(df["category"].isin(categories))].to_dict(orient="records")

df_test = pd.DataFrame(test_data)
test_data = df_test[(df_test["category"].isin(categories))].to_dict(
    orient="records"
)

cluster_model = DetoxifyModel()
pred_scores = []
prompts = []
categories = []


for entry in data:
    prob = cluster_model.predict(entry["response"])
    pred_scores.append(prob)
    prompts.append(entry["response"])
    categories.append(entry["category"])
print(len(data))

k = len(severity_labels)
n = 3

# Creating our clusters made with the 378-response data to base our new dataset on

kmeans = KMeans(n_clusters=k, random_state=42).fit(pred_scores)
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
    {"prompt": prompts, "category": categories, "cluster": cluster_labels}
)
df_clusters["cluster"] = df_clusters["cluster"].map(cluster_to_severity)
count_per_cluster = (
    df_clusters[df_clusters["category"] == "safe"].groupby("cluster").size()
)

counts = pd.crosstab(df_clusters["cluster"], df_clusters["category"])
print(counts)


# knn = KNeighborsClassifier(n_neighbors=n)
# mapped_cluster_labels = pd.Series(cluster_labels).map(cluster_to_severity)
# mapped_cluster_ids = mapped_cluster_labels.map(label_to_index)
# knn.fit(y_pred, mapped_cluster_ids)

# print(mapped_cluster_labels)

# y_pred_id_test = knn.predict(y_pred_test)


# y_pred_label_test = [severity_labels[id] for id in y_pred_id_test]
# print("Predicted severity:", y_pred_label_test)
# print("True severity:", y_true_label_test)







# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(y_pred)

# predX_pca = pca.transform(y_pred_test)

# PC1 = X_pca[:, 0]
# PC2 = X_pca[:, 1]
# scalePCA = 1

# fig, ax1 = plt.subplots(figsize=(10, 8))
# # plot per cluster.
# indexes = []
# cluster_scatters = []
# for i in range(k):
#     idx = np.where(i == cluster_labels)[0]
#     scatter = ax1.scatter(PC1[idx], PC2[idx], marker=".", label=cluster_to_severity[i])
#     indexes.append(idx)
#     cluster_scatters.append(scatter)

# pred_scatter = ax1.scatter(predX_pca[:, 0], predX_pca[:, 1], marker="X", c="black")

# cursor = mplcursors.cursor(pred_scatter, hover=True)


# @cursor.connect("add")
# def on_hover(sel):
#     # cluster_idx = cluster_scatters.index(sel.artist)
#     # true_idx = indexes[cluster_idx][sel.index]
#     index = sel.index
#     x, y = sel.target
#     report = (
#         f"Coordinates: ({x:.2f}, {y:.2f}) - Index: {index}\n"
#         f"Prompt: {prompts_test[index]}\n\n"
#         f"Expected Category: {category_test[index]}\n"
#         f"Predicted Severity Score: {y_pred_test[index]}\n\n"
#         f"Expected Severity Label: {y_true_label_test[index]}\n"
#         f"Predicted Severity Label: {severity_labels[y_pred_id_test[index]]}"
#     )
#     print(f"{report}")
#     sel.annotation.set(
#         text=report,
#         bbox=dict(fc="white", alpha=0.8),
#         wrap="true",
#     )


# ax1.set_title("Severity Levels: Report of Abuse VS Safe")
# ax1.set_xlabel(f"PCA 1")
# ax1.set_ylabel(f"PCA 2")
# plt.legend()
# plt.show()
