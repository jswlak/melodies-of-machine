#Convert raw text -> tokens (numbers for model input)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello Hugging Face!", return_tensors="pt")
print(tokens)
