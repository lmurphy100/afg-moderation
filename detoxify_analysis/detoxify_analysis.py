import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from weave import Model, Evaluation
import plotly.express as px
import mplcursors
import weave
from weave import Evaluation, Model
from mplcursors import cursor


weave.init('detoxify_severity_classification')


hf_model_name = 'unitary/toxic-bert'
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(hf_model_name) # Actual model


labels = ['toxicity','severe_toxicity','obscene','threat','insult','identity_attack']
severity_labels = ['N/A','Informational','Questionable','Urgent']

y_true = []  # Exisiting toxicity scores expected from the VALIDATION data. We can weight these. 
y_pred = []  # What toxicity scores are predicted from OUR data
y_pred_labels = [] # What severity score our model predicted from OUR data
prompts = [] # Prompts from OUR data

weightings = np.array([1, 1, 1, 1, 1, 1]) # Severe Toxicity, Obscene, Threat matter more
# Non linear, so these are the exponents 


# Get true scores in the validation set, for clustering



df = pd.DataFrame(pd.read_csv('./afg-moderation/detoxify_analysis/copy.csv'))   # Load the validation data
val_data = df.to_dict()

# Get true scores in the validation set, for clustering
for idx in range(len(df)): 
    values = []
    for label in labels:
        values.append(val_data[label][idx])
    values *= weightings
    y_true.append(values) 


'''
Create the Initial Clusters with KMeans. Get the Cluster Centroids
'''
y_true = np.array(y_true)

kmeans = KMeans(n_clusters=4, random_state=5).fit(y_true) 
kmean_counts = kmeans.predict(y_true)
clusterCount = np.bincount(kmean_counts)
print(clusterCount)

kmeans_labels = kmeans.labels_
centroids = kmeans.cluster_centers_



sorted_indices = np.argsort(centroids, axis=0)
sorted_centroids = np.take_along_axis(centroids, sorted_indices, axis=0)


'''
Detoxify model
'''
class DetoxifyModel(Model):  # Class just to process/predict input. 
    # Inputs a prompt, outputs multiple probabilities. 
    @weave.op()
    def predict(self, question: str) -> dict:   #: str just clarifies it is a stirng
        inputs = tokenizer(question, return_tensors="pt") # 2 arguments
        output = hf_model(**inputs)
        predictions = torch.sigmoid(output.logits)  # Converts logits to probabilities for multiple labels 
        predictions = predictions.detach().numpy().flatten()
        weighted = predictions * weightings
        #rescaled = 1/ (1 + np.exp(-weighted))
        return {'generated_scores':weighted,
                'predicted_severity':classify_severity(sorted_centroids, weighted)}
'''

'''
@weave.op()
def classify_severity(sorted_centroids, pred_point):  # Outputs index 
    pred_point = np.array(pred_point)
    
    # Originally using euclidian distance. Cosine approach is at the bottom 
    distances = []
    for cen in sorted_centroids:
        distances.append(distance.euclidean(pred_point, cen))
    return np.argmin(distances)

@weave.op()
def accuracy(expected: str, output: dict) -> dict:   # Put the return dict value in output.   Takes in 1 expected
  """
  expected: correct label the other dataset expects
  output: dict format if the model is correct
  """
  predicted_severity = [output['generated_label']] # Converts to one of the 5 content koala_labels

  return {'match_severity': predicted_severity == expected,
          'predicted_severity':output['predicted_severity'],
          'scores:':output['generated_scores'],
          }   # match: True

  # Start analyzing our responses.   
with open('./afg-moderation/responses.json', 'r') as file:    # Load our data
    data = json.load(file)  # convert to dict
    
'''
Weave Generation Report
'''

model = DetoxifyModel()   
evaluation = Evaluation(dataset=data, 
                        scorers=[accuracy],
                        preprocess_model_input=lambda row: 
                            {'question': row['question'],
                             'expected': row['expected']} 
) 


# # Run Detoxify against our Dataset prompts
# for prompt in prompts:
#     pred_point = dt_model.predict(prompt)
    
#     print('\nPrompt: ',prompt)
#     for i in range(len(labels)):        # Display labels corresponding to the toxicity score.   
#         print(f"{labels[i]}: {pred_point[i]:.4f}")
#     label = severity_labels[classify_severity(sorted_centroids, pred_point)]
    

#     y_pred.append(pred_point)
#     y_pred_labels.append(label)
    
#     print(pred_point)
#     print('Predicted Severity: ',label)

# sil_score = silhouette_score(y_true, kmeans_labels) # Quality of clusters. How well-separated and cohesive the clusters are, 
# print('Silhouette Score: ',sil_score)
# # 0.756

# '''
# PCA can
# - 
# - Tell us which features contribute to ethier axis
# - Tell us how accurate the 2D Graph is. In this case, reducing to 2 dimensions preserves 90% of the data variation. 
# '''

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(y_true)
# # PC1, PC2 preserve 90% of the variance. PC1, PC2, PC3 preserve 95% of the variance. 
# variances_axis = pca.explained_variance_ratio_
# loadings = pca.components_
# print(f" Variances: {variances_axis}")
# print(f"Loadings for Component: {loadings}")

# y_pred = np.array(y_pred)
# y_pred_pca = pca.transform(y_pred)
# y_pred_pca = np.array(y_pred_pca)

# print('Y predicted PCA: ',y_pred_pca)
# print('Y predicted Labels',y_pred_labels)

# PC1 = X_pca[:, 0]  # 1 dimensional, column one 
# PC2 = X_pca[:, 1] 
# scalePCA = 1

# fig, (ax1) = plt.subplots()  #ax1 - original data,   #ax2 - our data plotted on it
# scatter = ax1.scatter(x = PC1, 
#             y = PC2, 
#             c=kmeans.labels_, 
#             marker='.',)

# for i, feature in enumerate(labels):
#     ax1.arrow(0, 0,
#              dx =loadings[0, i]*scalePCA, 
#              dy =loadings[1, i]*scalePCA,
#              color='red') # Plot arrows
#     ax1.text(x=loadings[0, i]*(scalePCA),
#             y=loadings[1, i]*(scalePCA), 
#             s=feature, color='black')

# ax1.set_title("PCA Analysis with 3 Clusters")
# ax1.set_xlabel(f'PCA 1 - {(variances_axis[0]*100):.2f}%')
# ax1.set_ylabel(f'PCA 2 - {(variances_axis[1]*100):.2f}%')
# legend = ax1.legend(*scatter.legend_elements(), title='Clusters',loc='lower left')

# # Plot predictions as red dots
# predScatter = ax1.scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], 
#            color='red', marker='X', s=30, label='Predictions')





# cursor = mplcursors.cursor(predScatter, hover=True)

# @cursor.connect('add')
# def on_hover(sel):
#     index = sel.index
#     x, y = sel.target
#     sel.annotation.set(text=f"Coordinates: ({x:.2f}, {y:.2f}) - Index: {index}\n Prompt: {prompts[index]}\n\nSeverity Level: {y_pred_labels[index]}", 
#                        bbox=dict(fc='white', alpha=0.8), 
#                        wrap='true')    

# # BiPlot
# # scalePC1 = 1.0/PC1.max() - PC1.min()   # Min/Max scales for PC1
# # scalePC2 = 1.0/PC2.max() - PC2.min()

# # example_6d = example_6d.reshape(1, -1)
# # example_2d = pca.transform(example_6d).flatten()

# plt.show()
# #plt.tight_layout()


#    # - Cosine similarity approach 
#     # for cen in sorted_centroids:
#     #     cen = np.array(cen)
#     #     similar_direction = (np.dot(pred_point, cen)/(np.linalg.norm(pred_point) + np.linalg.norm(cen)))
#     #     distances.append(similar_direction)



