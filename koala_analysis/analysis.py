
import json
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
label_encoder = LabelEncoder()

with open('results.json', 'r') as file:
    data = json.load(file)  # convert to dict
    
y_pred = []
y_true = []
class_labels = ['hate speech','report of abuse','safe','sexual content','violence']


label_encoder.fit(class_labels)

for entry in data:
    prompt_number = entry['inputs']['example']['prompt_number']
    accuracy = entry['output']['scores']['accuracy']
    y_true.append(accuracy['true_label'])
    y_pred.append(accuracy['mapped_label'])
    
    print(f"Predicted: {accuracy['mapped_label']} - Expected {accuracy['true_label']} - {prompt_number}")

y_true_encoded = label_encoder.transform(y_true)
y_pred_encoded = label_encoder.transform(y_pred)


cm = confusion_matrix(y_true_encoded, y_pred_encoded)
print(classification_report(y_true_encoded, y_pred_encoded, target_names=class_labels))

# Set up the plot
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)  # Larger font for better readability

# Customize heatmap
heatmap = sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    cbar=False,
    linewidths=0.5,
    linecolor='lightgray',  
    xticklabels=class_labels, 
    yticklabels=class_labels,
    annot_kws={'size': 12}  
)

plt.xlabel('Predicted Label', fontsize=14, labelpad=15)
plt.ylabel('True Label', fontsize=14, labelpad=15)
plt.title('Confusion Matrix\n(Student Response Moderation)', fontsize=16, pad=20)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
