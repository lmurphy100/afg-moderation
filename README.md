# MODEL-BASED EVALUATIONS 

-[Koala Evaluation](./weave_koala/koala.md)
-[LLamaGuard Evaluation](./weave_llamaguard/llamaguard.md)
-[Detoxify Evaluation](./weave_detoxify/detoxify.md)

# HOW TO RUN 

Running each model and generating their weave evaluations is simple. 

To ensure there are no conflicts. We use a virtual enviroment to isolate dependecies for the project from the global system

To create the virtual enviroment:
`python venv .venv`
To activate the virtual enviroment:
`./venv/Scripts/Activate.ps1`

Now you can run the 3 generation scripts:


## Llama Guard Model

LLama-guard is a guard-rail model finetuned from LLama 2 with new data. The model is not light-weight, consisting of over 8 billion parameters.

LLama-Guard is a gated model, meaning you must submit a usage request in the huggingface website. Once accessed, you can 


To generate the weave report, run `llama_test.py` with 
`python llama_test.py` on your terminal. 
Weave will individually evaluate your entries and upload traces to the W&B Weave UI. 

After the evaluation, you will be prompted to login to your weave dashboard. 



Through 'Evaluations', you can see the current evaluation and compare previous ones. 
Remove any unnecessary columns. 



