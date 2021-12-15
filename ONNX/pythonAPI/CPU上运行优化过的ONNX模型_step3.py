# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/15 11:17
# software: PyCharm

"""
文件说明：
ONNX Runtime 为支持不同的硬件加速ONNX models，引入了一个可扩展的框架，称为Execution Providers(EP)，集成硬件中特定的库。
在使用过程中只需要根据自己的真实环境和需求指定InferenceSession中的providers即可，
比如如果想要用CPU那么可以如此创建会话：session =InferenceSession(model_path,options,providers=['CPUExecutionProvider'])。

优化后的graph可能包括各种优化，如果想要查看优化后graph中一些更高层次的操作
(例如EmbedLayerNormalization、Attention、FastGeLU)可以通过比如Netron等可视化工具查看。
    
"""

from os import environ
from psutil import cpu_count

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))  # OMP 的线程数
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

from contextlib import contextmanager
from dataclasses import dataclass
from time import time
from tqdm import trange


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session



from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese") # 使用 Pytorch 模型的字典
cpu_model = create_model_for_provider("onnx/bert-base-chinese.opt.onnx", "CPUExecutionProvider") # 使用 优化过的 onnx

# Inputs are provided through numpy array
model_inputs = tokenizer("大家好, 我是卖切糕的小男孩, 毕业于华中科技大学", return_tensors="pt")
inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}

# Run the model (None = get all the outputs)
sequence, pooled = cpu_model.run(None, inputs_onnx)

# Print information about outputs
print(f"Sequence output: {sequence.shape}, Pooled output: {pooled.shape}")


