import os

from minbpe.minbpe.basic import BasicTokenizer

filename = os.path.join(os.path.dirname(__file__), "kings.txt")
with open(filename, "r") as file:
    text = file.read()

extra_vocab = 4000
tokenizer = BasicTokenizer()
tokenizer.train(
    text, 256 + extra_vocab
)  # 256 are the byte tokens, then do 10000 merges
tokenizer.save(os.path.join(os.path.dirname(__file__), "kings_tokenizer"))
