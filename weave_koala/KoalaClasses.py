from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import wandb
import weave
from weave import Model, Scorer
import torch
import pandas as pd
import json
import numpy as np
import os
print('CURRENT DIR IN FILE: ', os.getcwd())

with open("./config.json", "r") as f:
    config = json.load(f)
koala2cat = config["koala2cat"]
cat_labels = config["category_labels"]

y_pred = []
y_true = []
y_pred_binary = []
y_true_binary = []
y_pred = []

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  
print('Using ',device)

tokenizer = AutoTokenizer.from_pretrained("KoalaAI/Text-Moderation")
hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path = "KoalaAI/Text-Moderation").to(device)

koala_labels = koala_labels = ["H", "H2", "HR", "OK", "S", "S3", "SH", "V", "V2"]

class KoalaModel(Model):  # Class just to process/predict input.

    # Inputs a Prompt to output a label.
    @weave.op(name='accuracy_scorer')
    def predict(self, response: str) -> dict:  #: str just clarifies it is a stirng
        inputs = tokenizer(response, return_tensors="pt").to(device)
        output = hf_model(**inputs)  # there are 2 arguments
        logits = output.logits
        probabilities = logits.softmax(    # Tensor
            dim=-1
        ).squeeze() 
        probabilities = probabilities.cpu().detach().numpy()  # Convert from tensor to array
        output = koala_labels[np.argmax(probabilities)]
        return {
            "pred": output}
        # # here's where you would add your LLM call and return the output
    def make_confusion_matrix(self):
        bin_labels = ["safe", "unsafe"]

        y_pred_encoded = LabelEncoder().fit(cat_labels).transform(y_pred)
        y_true_encoded = LabelEncoder().fit(cat_labels).transform(y_true)

        y_predbin_encoded= LabelEncoder().fit(bin_labels).transform(y_pred_binary)
        y_truebin_encoded = LabelEncoder().fit(bin_labels).transform(y_true_binary)

        report_dict = classification_report(
            y_pred=y_pred_encoded,
            y_true=y_true_encoded,
            target_names=cat_labels,
            output_dict=True,
        )
        category_report_table = []
        for cat in report_dict:
            metrics = report_dict[cat]
            if cat != "accuracy":
                line = [
                    cat,
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1-score"],
                    metrics["support"],
                ]
                category_report_table.append(line)
        print("Table: ", category_report_table)

        category_cm = wandb.plot.confusion_matrix(
            preds=y_pred_encoded,
            y_true=y_true_encoded,
            probs=None,
            title="Confusion Matrix",
            class_names=cat_labels,
        )
        report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
        df = pd.DataFrame(columns=report_columns, data=category_report_table)

        wandb.run.log(
            {
                "Confusion Matrix (Category)": category_cm,
                "Classification Report (Category)": df,
            }
        )
        binary_cm = wandb.plot.confusion_matrix(
            preds=y_predbin_encoded,
            y_true=y_truebin_encoded,
            probs=None,
            title="Safe vs Unsafe Confusion Matrix",
            class_names = bin_labels
        )
        wandb.log({"Confusion Matrix (Binary)": binary_cm})
        wandb.finish()


class KoalaCategoryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        label = koala2cat[
            output["pred"]
        ]  # Converts to one of the 5 content koala_labels
        print(output['pred'], label, category)
        y_true.append(category)
        y_pred.append(label)
        return {
            "mapped_category": label,
            "match_category": label == category,
        }  

class KoalaBinaryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        
        label_true = "safe" if category == "safe" else "unsafe"
        label_pred = "safe" if output["pred"] == "OK" else "unsafe"

        y_true_binary.append(str(label_true))
        y_pred_binary.append(str(label_pred))
        return {
            "mapped_binary": label_pred,
            "match_binary": label_pred == label_true,
        } 