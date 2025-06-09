from weave_models.DetoxifyClasses import DetoxifyModel, SeverityLevelScorer
from weave_models.KoalaClasses import KoalaModel, AccuracyScorer
#from weave_models.LLamaGuardClasses import LLamaGuardModel
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
severity_labels = ["N/A", "Informational", "Questionable", "Urgent"]
weightings = np.array([1, 1, 1, 1, 1, 1])

severity_scorer = SeverityLevelScorer()  # Detoxify Scorer
accuracy_scorer = AccuracyScorer()  # Koala Scorer
koala_model = KoalaModel()

results_data = []

for entry in data:
    koala_pred = koala_model.predict(entry['response'])
    koala_score = accuracy_scorer.score(
        expected=entry["category"], output=koala_pred)
    temp = temp.update(koala_pred)
    temp = temp.update(koala_score)
    
    results_data.append(temp)

results_df = pd.DataFrame(results_data)  # Convert results to a DataFrame

print(results_df)
results_df.to_csv("results.csv", mode='w+',index=False)  # Save results to a CSV file




#detoxify_model = DetoxifyModel(weightings, severity_labels, y_true)

# val_data = pd.DataFrame(pd.read_csv("./datasets/detoxify_validation.csv")).to_dict()
# y_true = []
# for idx in range(len(val_data)):
#     values = []
#     for label in labels:
#         values.append(val_data[label][idx])
#     weighted_values = values*np.array(weightings)
#     y_true.append(weighted_values)
    
# with open("./datasets/responses-375.json", "r") as file:
#     data = json.load(file)  # convert to dict
# detoxify_pred = detoxify_model.predict(entry['question'])
    # detoxify_scorer = severity_scorer.score(
    #     expected_severity_level=entry["expected_severity_level"],
    #     output=detoxify_model.predict(entry["question"]),
    # )