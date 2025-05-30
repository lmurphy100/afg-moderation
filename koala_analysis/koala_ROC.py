
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/Text-Moderation") # Actual model

import json
import pandas as pd
import duckdb
import torch
import numpy as np
import weave
from weave import Model
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

koala_labels = ["H", "H2", "HR", "OK", "S", "S3","SH","V","V2"]
content_labels = ["hate speech", "report of abuse", "safe", "sexual content", "violence"]

conversion = {
    "H":"hate speech",
    "H2":"hate speech",
    "HR":"report of abuse",
    "OK":"safe",
    "S":"sexual content",
    "S3":"report of abuse",
    "SH":"violence",
    "V":"violence",
    "V2":"violence",
}

y_pred = []
y_pred_aggregated = []
y_true = []

'''
Same structure as other classes: But outputs raw probabities, not a label. 
'''
class KoalaModel(Model):  # Class just to process/predict input. S

    def predict(self, question: str) -> dict: 
        inputs = tokenizer(question, return_tensors="pt") # 2 arguments
        output = hf_model(**inputs) # there are 2 arguments
        logits = output.logits
        probabilities = logits.softmax(dim=-1).squeeze()  # Normalizes the logits between 0-1 and flattens them
        probabilities = probabilities.detach().numpy() # Convert from tensor to array
        return probabilities   # Returns 9 probabilities

model = KoalaModel()
with open('afg-moderation/responses.json', 'r') as file:
    data = json.load(file)  # convert to dict
for entry in data:
    
    expected = entry['expected']
    prompt = entry['question'] # Label (Doesn't need to be One Hot Encoded)
    prediciton = model.predict(prompt) # Probailities in a 1D Array
    y_pred.append(prediciton)
    y_true.append(expected)

# y_pred
'''
Aggregated Probabilities: Because there is a class mismatch with Koala, we need to sum probabilities that correlate 
with the conversion table above. Then feed it into the Receiver Operating Characteristic (ROC) curve. 
'''
def aggregate_probabilities(koala_probs):  # For one entry
    
    prob_sum = 0.0
    content_probs = np.zeros(5)
    for koala_label, prob in zip(koala_labels, koala_probs):   # Aggregate probabilities
        converted_label = conversion[koala_label]
        converted_label_idx = content_labels.index(converted_label)
        content_probs[converted_label_idx] += prob
        prob_sum += prob
        print(koala_label, prob, converted_label_idx)
    return content_probs 
    
    
for i in range(len(y_pred)):
    y_pred_aggregated.append(aggregate_probabilities(y_pred[i]))

print('Aggregated Probabilities: ',y_pred_aggregated)
y_pred_aggregated = np.array(y_pred_aggregated)

y_true = np.array(y_true)

'''
Because ROC is largely a binary task, We use the Label Binarizer to encode the labels and feed in the 
encoded true labels, and aggregated predicted probabiltites. 

We use Micro-Averaging because our dataset classes are widely imbalanced. The Area Under Curve (AOC) indicates
how well our model distinguses between classes and the overall performance. 
'''
label_binarizer = LabelBinarizer().fit(y_true)
y_onehot_test_true = label_binarizer.transform(y_true) # 1D array 
print(y_onehot_test_true.shape) 
print(label_binarizer.classes_)
# (NUM SAMPLES, NUM LABELS)

class_of_interest = 'safe' 
label_binarizer.transform([class_of_interest])  
class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
print(class_id) # Should be 2

micro_display = RocCurveDisplay.from_predictions(
    y_true = y_onehot_test_true.ravel(), # only column of class id
    y_pred = y_pred_aggregated.ravel(), # Probability scores
    name="Micro-Average (One Vs Rest)",
    color='orange',
    plot_chance_level = True,
    despine = True
)
plt.title('Micro-Averaged ROC Curve for 5 classes')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

