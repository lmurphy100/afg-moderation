import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration, AutoTokenizer
from huggingface_hub import login
import numpy as np
import json
import weave
from weave import Model, Scorer
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import pandas as pd
from dotenv import load_dotenv
import os
import wandb

load_dotenv()  # Load env vars

with open("./config.json", "r") as f:
    config = json.load(f)

login(os.getenv("HF_TOKEN"))  # Login to HuggingFace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_id = "meta-llama/Llama-Guard-4-12B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, load_in_8bit=True
)
print(f"Using device: {device}")

llama2cat = config["llama2cat"]
cat_labels = config["category_labels"]

y_pred = []
y_true = []
y_pred_binary = []
y_true_binary = []


class LGModel(Model):
    @weave.op()
    async def predict(self, response: str) -> dict:
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": response,
                    }
                ],
            }
        ]

        inputs = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, pad_token_id=0)
            prediction = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )[0].split()
        return {"llama_pred": prediction}

    def make_confusion_matrix(self):
        bin_labels = ["safe", "unsafe"]

        y_pred_encoded = LabelEncoder().fit(cat_labels).transform(y_pred)
        y_true_encoded = LabelEncoder().fit(cat_labels).transform(y_true)

        y_predbin_encoded = LabelEncoder().fit(bin_labels).transform(y_pred_binary)
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
            class_names=bin_labels,
        )
        wandb.log({"Confusion Matrix (Binary)": binary_cm})
        wandb.finish()


class LGCategoryScorer(Scorer):
    @weave.op()
    def score(self, category: str, output: dict) -> dict:
        y_predicted = ""
        llama_pred = output["llama_pred"]
        predicted = llama_pred[0] if llama_pred[0] == "safe" else llama_pred[1]
        print("Predicted: ", predicted)

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
