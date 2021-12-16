# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 12:48
# software: PyCharm

"""
文件说明：
    
"""
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from onnxruntime import InferenceSession, SessionOptions

options = SessionOptions()
session = InferenceSession("./embeddings.onnx", options)

tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
tokens = tokenizer(["I am happy", "I am glad"], return_tensors="np")

outputs = session.run(None, dict(tokens))[0]
print(outputs)
print(cosine_similarity(outputs))
