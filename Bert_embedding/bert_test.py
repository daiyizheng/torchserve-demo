# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/6 9:24
# software: PyCharm

"""
文件说明：
    
"""
pretrained_model_name = "bert-base-chinese"
torchscript = True
do_lower_case = True
NEW_DIR = "./"
import os
import torch
from transformers import BertModel, AutoTokenizer, BertConfig


device = torch.device('cpu')
config = BertConfig.from_pretrained(pretrained_model_name,torchscript=torchscript)
model = BertModel.from_pretrained(pretrained_model_name, config=config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
dummy_input = ["做基因检测采集什么样本检测结果最准确"]
inputs = tokenizer.batch_encode_plus(dummy_input,max_length=50,pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
# model.to(device).eval()
# traced_model = torch.jit.trace(model, (input_ids, attention_mask))
# torch.jit.save(traced_model,os.path.join(NEW_DIR, "traced_model.pt"))
a = model(input_ids, attention_mask)
print(a[1].detach().numpy())