# nanoLLMs

Implementation of three different type of LLMs that are [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) , [Mamba](https://arxiv.org/pdf/2312.00752) and [xLSTM](https://arxiv.org/pdf/2405.04517).

Some examples of Mamba models are  : xLSTM ....

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

You may want to run small LLM to generate Shakespeare-like text, please check out the notebook `nanoLLMs/notebooks/shakespeare_char.ipynb` that trains GPT, Mamba and xLSTM models from scratch. 

## LLM Pre-training on Wikipedia data

The goal of this experiment is to train a rather large LLM (1B) on data scrapped from Wikipedia. French Kings Wikipedia pages were scrapped in order to create a text training dataset. After that, a BPE tokenizer was trained with a 4k vocabulary (thanks to https://github.com/karpathy/minbpe).

To do it on your own machine, please run (this should take a couple of mins): 
```bash 
python python data/wiki_french_kings/scrap_wikipedia.py
python python data/wiki_french_kings/train_tokenizer.py
python python data/wiki_french_kings/prepare.py
```



