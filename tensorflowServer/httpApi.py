# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 13:47
# software: PyCharm

"""
文件说明：
    
"""
from tensorflowServer import tokenization
import json


def process_input(texts, max_len):
    def tokenize_input(text):
        tokenizer = tokenization.FullTokenizer("./chinese_L-12_H-768_A-12/vocab.txt")
        tokens = tokenizer.tokenize(text)
        text_ids = tokenizer.convert_tokens_to_ids(tokens)
        while len(text_ids) < max_len:
            text_ids.append(0)
        return text_ids

    input_ids = []
    for text in texts:
        text_ids = tokenize_input(text)
        input_ids.append(text_ids)

    return input_ids


import numpy as np
import requests


def predict_http(input_texts):
    SERVER_URL = "http://10.0.30.2:8526/v1/models/bert/versions/1:predict"

    input_ids = process_input(input_texts, max_len=128)
    input_ids = np.array(input_ids).tolist()
    input_data = {"input_ids": input_ids}

    request_data = {"signature_name": "serving_default", "instances": input_data}
    request_data = json.dumps(request_data)

    response = requests.post(SERVER_URL, data=request_data)
    result = response.json()
    pred_value = result['outputs']

    return pred_value

if __name__== "__main__":
    pred_value = predict_http(["你好啊","阿萨大大"])