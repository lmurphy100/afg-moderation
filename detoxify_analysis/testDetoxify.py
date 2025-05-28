import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import weave
from weave import Model, Evaluation

#weave.init('detoxify_severity_classification')

hf_model_name = 'unitary/toxic-bert'
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name) # Actual model


labels = ['toxicity','severe_toxicity','obscene','threat','insult','identity_attack']
y_true = []  # For validation data
y_pred = []  # For our data 

prompts = []

df = pd.DataFrame(pd.read_csv('./detoxify_analysis/copy.csv'))   # Load the validation data
val_data = df.to_dict()

with open('./responses.json', 'r') as file:    # read our data
    data = json.load(file)  # convert to dict
for entry in data:
    prompts.append(entry['question'])

# Get true scores in the validation set
for idx in range(len(df)): 
    values = []
    for label in labels:
        values.append(val_data[label][idx])
    y_true.append(values)

labels = ['toxicity', 'severe toxicity', 'obscene','threat','insult','identity attack']
severity_labels = ['Informational','Questionable','Urgent']


class DetoxifyModel(Model):  # Class just to process/predict input. 
    # Inputs a prompt, outputs multiple probabilities. 
    #@weave.op()
    def predict(self, question: str) -> dict:   #: str just clarifies it is a stirng
        inputs = tokenizer(question, return_tensors="pt") # 2 arguments
        output = hf_model(**inputs)
        predictions = torch.sigmoid(output.logits)  # Converts logits to probabilities for multiple labels 
        predictions = predictions.detach().numpy().flatten()
        return predictions
     

def classify_severity(sorted_centroids, pred_point):  # Outputs index 
    distances = []
    for cen in sorted_centroids:
        distances.append(distance.euclidean(pred_point, cen))
    return np.argmin(distances)


dt_model = DetoxifyModel()   
# evaluation = Evaluation(dataset=data, 
#                         scorers=[classify_severity],
#                         preprocess_model_input=lambda row: 
#                             {'question': row['question'],
#                              'expected': row['expected']} 
# ) 

# results = asyncio.run(evaluation.evaluate(model))  # Forwards results to website/database


# Create the Clusters with KMeans. 

y_true = np.array(y_true)
kmeans = KMeans(n_clusters=3, random_state=5).fit(y_true) 
kmeans_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

sorted_indices = np.argsort(centroids, axis=0)
sorted_centroids = np.take_along_axis(centroids, sorted_indices, axis=0)


for prompt in prompts:
    pred_point = dt_model.predict(prompt)
    print('\nPrompt: ',prompt)
    for i in range(len(labels)):        
        print(f"{labels[i]}: {pred_point[i]:.4f}")
    label = severity_labels[classify_severity(sorted_centroids, pred_point)]
    print('Predicted Severity: ',label)

#sil_score = silhouette_score(y_true, kmeans_labels) # Quality of clusters. How well-separated and cohesive the clusters are, 
# 0.756


# Do PCA to see a visualization

#X_pca = PCA(n_components=2).fit_transform(y_true)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, marker='.')

# plt.scatter(actual_toxicity, actual_threat)
# plt.xlabel('True Toxicity')
# plt.ylabel('True Threat')
#plt.show()




# for key, val in res.items():
#     print(f"{key} - {val:.2f}")
