# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 15:31
# software: PyCharm

"""
文件说明：
    
"""
import torch
from transformers import BertTokenizerFast, BertModel
device =  torch.device('cpu')
# Load Pytorch model
torch_model = "bert-base-chinese"
model = BertModel.from_pretrained("bert-base-chinese").to(device)

tokenizer = BertTokenizerFast.from_pretrained(torch_model)

model.eval()

inputs = tokenizer("大家好, 我是卖切糕的小男孩, 毕业于华中科技大学", return_tensors="pt")

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
token_type_ids = inputs["token_type_ids"].to(device)
# print(model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids))
torch.onnx.export(
    model,
    #model.module  [===>> if dataparallel-model ==>> see above commented line 20]

    #we have 3 inputs so  name them here -- ordering is important
    (input_ids, attention_mask, token_type_ids),

    #export it to model.onnx
    'model.onnx',

    #in same order as we wrote inputs
    input_names = ['input_ids','attention_mask', 'token_type_ids'],

    #anything that you want
    output_names = ['output'],
    opset_version=10,
    #we know that batch size is dynamic as axis of batch size is index 0 so ==>>> {0:"batch_size"} ==>> for each inputs and outputs
    dynamic_axes = {
        'input_ids':{0:"batch_size"},
        'attention_mask':{0:"batch_size"},
        'token_type_ids':{0:"batch_size"},
        'output':{0:"batch_size"},
    }
)