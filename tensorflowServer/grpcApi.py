# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 14:41
# software: PyCharm

"""
文件说明：
    
"""


import grpc
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflowServer import tokenization
import json
host = "10.0.30.2"
port = "8526"

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



# initialize grpc channel
channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))

def predict_grpc(input_texts, max_len=128):

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    result = []

    def predict_request(input_ids):
        """模型预测请求."""
        input_ids = np.array(input_ids)
        input_tensor = tf.make_tensor_proto(input_ids, shape=input_ids.shape, dtype=tf.int32)
        try:
            request = predict_pb2.PredictRequest()
            request.inputs["input_ids"].ParseFromString(input_tensor.SerializeToString())
            request.model_spec.name = "bert"
            request.model_spec.signature_name = "serving_default"
            response = stub.Predict(request, 50)
            _result = tf.make_ndarray(response.outputs["outputs"]).tolist()
            result.extend(_result)
        except Exception as e:
            print(e)

    input_ids = process_input(input_texts, max_len)
    predict_request(input_ids)

    return result

if __name__== "__main__":
    pred_value = predict_grpc(["你好啊","阿萨大大"])