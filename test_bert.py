from bigdl.llm.transformers import AutoModelForMaskedLM
from transformers import BertTokenizer
import torch
from torch.optim import AdamW
import torch.nn as nn

# Load the BERT tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Example input text
input_text = "Here is some example text to encode."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Forward pass through BERT
outputs = model(input_ids,labels=input_ids)
print(outputs.__dict__.keys())
loss=outputs.loss

# Optimize
optimizer = AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss after one step of optimization:", loss.item())

