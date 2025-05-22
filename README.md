# nanoLLMs

Implementation of three different type of LLMs that are [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) , [Mamba](https://arxiv.org/pdf/2312.00752) and [xLSTM](https://arxiv.org/pdf/2405.04517).

Some examples of Mamba models are Mistral's [Codestral](https://mistral.ai/news/codestral-mamba)  [Falcon series](https://falconllm.tii.ae/tii-releases-first-sslm-with-falcon-mamba-7b.html), xLSTM's model from [NXAI](https://www.nx-ai.com/).

All models were trained on Kaparthy's [educational example](https://github.com/karpathy/nanoGPT) on Shakespeare text.

Model implementation was done and partially inspired by : 

- GPT2 : https://github.com/karpathy/nanoGPT
- MAMBA : https://github.com/alxndrTL/mamba.py
- xLSTM : https://github.com/NX-AI/xlstm

## Installation 
Please run, the following command to install relevant dependencies:
```bash 
pip install .
```
To prepare the dataset, please run (which is the same command as in [nanoGPT](https://github.com/karpathy/nanoGPT)).
```bash 

python data/shakespeare_char/prepare.py
```

You may want to run small LLM to generate Shakespeare-like text, please check out the notebook `notebooks/shakespeare_char.ipynb` that trains GPT, Mamba and xLSTM models from scratch. 




