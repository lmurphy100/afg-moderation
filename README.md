# Purpose

For proper evaluation


-[Overall Analaysis](ANALYSIS.md)

# How to Run

To ensure there are no conflicts. We use a virtual enviroment to isolate dependecies for the project from the global system

To activate the virtual enviroment:
`./venv/Scripts/Activate.ps1`

Now you can run the 3 generation scripts. 

Make sure you have a W&B Weave Account 
`config.json` 




## Koala Model



## Llama Guard Model

LLama-guard is a guard-rail model finetuned from LLama 2 with new data. The model is not light-weight, consisting of over 
8 billion parameters. Because of this, evaluations are slow and can be accelerated with a GPU instead of ultilizing the CPU. 

See
-[NVIDIA support](#cuda-installation)
-[AMD support](#roc-installation)

LLama-Guard is a gated model, meaning you must submit a usage request in the huggingface website. Once accessed, you can 


To generate the weave report, run `llama_test.py` with 
`python llama_test.py` on your terminal. 
Weave will individually evaluate your entries and upload traces to the W&B Weave UI. 

After the evaluation, you will be prompted to login to your weave dashboard. Through 'Evaluations', you can see the current evaluation and compare previous ones. 
Remove any unnecessary columns. 



After the evaluation, there is a csv generated explaining the whole thing. 



## Extras

### Cuda Installation

This is the preferred way to ultilize a GPU

Run `nvidia-smi` to display information about your GPU and look for CUDA Version 12.X. Find your NVIDIA toolkit for that version
and install it. Then, PyTorch will automatically detect and use your GPU for model inference, accelerating the process. 

### ROC Installation

This is an alternative way for AMD GPU's if you have one. 