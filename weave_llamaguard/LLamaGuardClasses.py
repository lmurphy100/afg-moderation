import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import numpy as np
import json
import weave
from weave import Model, Scorer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import wandb
from dotenv import load_dotenv
load_dotenv() # Load env vars


with open("./config.json", "r") as f:
    config = json.load(f)

login(os.getenv('HF_TOKEN')) # Login to HuggingFace
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model_id = "meta-llama/Llama-Guard-4-12B"
model_id = "meta-llama/Llama-Guard-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = ""
if(not torch.cuda.is_available):
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
else:
    hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, load_in_8bit=True
)

print("Model is using: ", hf_model.device)
llama2cat = config["llama2cat"]
cat_labels = config["category_labels"]
y_pred = []
y_true = []
y_pred_binary = []
y_true_binary = []


class LGModel(Model):
    @weave.op()
    async def predict(self, response: str) -> dict:
        # If predicted contains expected category, y_pred = category, if not, y_pred = first index on predicted dict

        print("Ran model")
        chat = [{"role": "user", "content": response}]
        input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt")
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output = hf_model.generate(
                input_ids=input_ids, max_new_tokens=5, pad_token_id=0
            )
        prompt_lense = input_ids.shape[-1]  # Size of sequence
        results = tokenizer.decode(
            output[0][prompt_lense:], skip_special_tokens=True
        ).split()  # extracts from size of input sequence to <eot>, which is our answer. puts into array
        print(f"Predicted {results}")
        return {"llama_pred": results}

    def make_confusion_matrix(self):
        bin_labels = ["safe", "unsafe"]

        y_pred_encoded = LabelEncoder().fit(cat_labels).transform(y_pred)
        y_true_encoded = LabelEncoder().fit(cat_labels).transform(y_true)

        y_predbin_encoded= LabelEncoder().fit(bin_labels).transform(y_pred_binary)
        y_truebin_encoded = LabelEncoder().fit(bin_labels).transform(y_true_binary)

        print(f"Pred encoded: {y_pred_encoded}")
        print(f"True encoded: {y_true_encoded}")

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


class LGCategoryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        y_predicted = ""
        llama_pred = output["llama_pred"]
        predicted = llama_pred[0] if llama_pred[0] == "safe" else llama_pred[1]
        print(predicted)

        if category in llama2cat[predicted]:
            y_predicted = category
        else:
            y_predicted = llama2cat[predicted][0]
        y_pred.append(str(y_predicted))
        y_true.append(str(category))
        return {
            "mapped_category": str(y_predicted),
            "match_category": y_predicted == category,
        }


class LGBinaryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        label_true = "safe" if category == "safe" else "unsafe"
        label_pred = "safe" if output["llama_pred"][0] == "safe" else "unsafe"

        print(label_true, label_pred)
        y_true_binary.append(str(label_true))
        y_pred_binary.append(str(label_pred))
        return {
            "mapped_binary": str(label_pred),
            "match_binary": label_pred == label_true,
        }

        # print(f"Predicted: {y_pred}\n\nActual: {y_true}")
        # print(f"Predicted: {y_pred_binary}\n\nActual: {y_true_binary}")
