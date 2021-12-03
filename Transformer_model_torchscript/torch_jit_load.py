# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/3 9:13
# software: PyCharm

"""
文件说明：
    
"""
import torch
from transformers import AutoTokenizer
text = "Bloomberg has decided to publish a new report on the global economy."

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = torch.jit.load("traced_model.pt", map_location=device)
print(type(text))
inputs = tokenizer.encode_plus(text,  max_length=150, pad_to_max_length=True, add_special_tokens=True, return_tensors='pt')
input_ids = inputs["input_ids"].to(device)
print("*************************input_ids***************************", input_ids)
attention_mask = inputs["attention_mask"].to(device)
model = model.to(device)
a = model(input_ids, attention_mask)
print(a)
