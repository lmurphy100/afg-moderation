from weave_models.DetoxifyClasses import DetoxifyModel
from weave_models.KoalaClasses import KoalaModel, AccuracyScorer
from weave_models.LLamaGuardClasses import LLamaGuardModel
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

labels = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
# Models
koala_model = KoalaModel()
llama_model = LLamaGuardModel()
# Scorers
accuracy_scorer = AccuracyScorer()  # Koala Scorer

with open("./afg-moderation/datasets/responses-375.json", "r") as file:
    data = json.load(file)  # convert to dict
results_data = []
idx = 0

def process_entry(entry):
    prompt = entry["response"]
    chat = [{"role": "user", "content": prompt}]
    temp = {}
    koala_pred = koala_model.predict(prompt)
    koala_score = accuracy_scorer.score(category=entry["category"], output=koala_pred)
    llama_pred = llama_model.predict(chat)
    temp.update(koala_pred) 
    temp.update(koala_score)
    temp.update(llama_pred)
    idx+=1
    print(idx)

    return temp

print(multiprocessing.cpu_count())  #22 cpus available
num_cpus = multiprocessing.cpu_count() / 2  # Use half of the available CPUs or 'treads'

with ThreadPoolExecutor(max_workers=num_cpus) as executor:
    results = list(executor.map(process_entry, data))




# results_df = pd.DataFrame(results_data)  # Convert results to a DataFrame
# print(results_df)
# results_df.to_csv("results.csv", mode="w+", index=False)  # Save results to a CSV file


# detoxify_model = DetoxifyModel(weightings, severity_labels, y_true)

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
