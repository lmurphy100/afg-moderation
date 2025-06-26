# WEAVE EVALUATION FRAMEWORK
This is a framework desgined to be configerable with your own dataset and categories. Mapping with LLamaGuard and Koala is
also configerable. 

# Model-Based Evaluations 

-[Koala Evaluation](./weave_koala/koala.md)
-[LLamaGuard Evaluation](./weave_llamaguard/llamaguard.md)
-[Detoxify Evaluation](./weave_detoxify/detoxify.md)

# How to Run

Running each model and generating their weave evaluations is simple. 

## Setting an Enviroment
### Python Virtual Enviroments
To ensure there are no conflicts. We use a virtual enviroment to isolate dependecies for the project from the global system
Make sure you are one directory above afg-open-response-safety folder

To create the virtual enviroment:
- `python -m venv venv` 
To activate the virtual enviroment:
- `./venv/Scripts/Activate.ps1`

Install dependacies:
cd into eval 
- `cd afg-open-response-safety/eval`
- `pip install -r requirements.txt`

### Enviroment Variables

Create a `login.env` file
Enter your API key and host-name to upload your weave evaluations. 

### Config 

This tool has a `config.json` file where you can

* Customize the category labels the confusion matrix has
* Change mapping schema for LlamaGuard and Koala. 
`
Now you are ready to run the 3 generation scripts.

## Koala Model
Koala-AI is a lightweight transformer model. (189 million parameters)

There is 1 weave script for Koala AI. 
* Run KoalaAI eval: `python koala_weave_generation.py`

- [Weave Basics](#weights-and-biases-basics)


## Llama Guard Model

LLama-Guard is a medium-sized LLM finetuned from LLama 2 with new data. LLamaGuard 3: (8.3 billion parameters), LLamaGuard 4: (12.1 billion parameters)

LLama-Guard is a gated model, meaning you must submit a usage request in the huggingface website. Once accessed, you can 

There are 2 weave scripts for LLamaGuard 3 and LLamaGuard 4 evaluations. 

* Run LLamaGuard 3 eval: `python llamaguard8b_weave.py`
* Run LLamaGuard 4 eval: `python llamaguard12b_weave.py`

- [Weave Basics](#weights-and-biases-basics)

## Detoxify Model

(IN PROGRESS)
Detoxify differs significantly from Koala and LLamaGuard because it is a regression-based model.
It is used primarily for clustering severity levels. 

PCA Analysis too. 


- [Weave Basics](#weights-and-biases-basics)


## Weights and Biases Basics. 
Weights and Biases (W&B) is the software used for streamlined analysis of models. 
While we host our evaluations on a private afg cloud, you can adjust enviroment variables in `ENV.env` to 
run it on our instance of weave or a seperate host-name. 

You will two components when the weave evaluation concludes: Model and Weave. 

### Model
Overall analysis of the model, 
* Confusion Matrix
* Other Classification Metrics (Precision, Recall, F1-Score)
* ROC Curve
In `Workspace`, you can see confusion matrix comparisons between model runs. 

### Weave
Used for comparing individual prompts 
Through `Evaluations`, you can see the current evaluation and compare previous ones as the baseline. This will show
you individual prompt analsysis between the various models. *(This was used specifically in LLamaGuard evaluation between LLamaGuard 3 and LLamaGuard 4)*



