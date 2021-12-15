# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/15 11:22
# software: PyCharm

"""
文件说明：
    
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


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)

@dataclass
class OnnxInferenceResult:
    model_inference_time: [int]
    optimized_model_path: str

## Pytorch
from transformers import BertModel

PROVIDERS = {
    ("cpu", "PyTorch CPU"),
    #  Uncomment this line to enable GPU benchmarking
    #    ("cuda:0", "PyTorch GPU")
}

results = {}

for device, label in PROVIDERS:

    # Move inputs to the correct device
    model_inputs_on_device = {
        arg_name: tensor.to(device)
        for arg_name, tensor in model_inputs.items()
    }

    # Add PyTorch to the providers
    model_pt = BertModel.from_pretrained("bert-base-chinese").to(device)
    for _ in trange(10, desc="Warming up"):
        model_pt(**model_inputs_on_device)

    # Compute
    time_buffer = []
    for _ in trange(100, desc=f"Tracking inference time on PyTorch"):
        with track_infer_time(time_buffer):
            model_pt(**model_inputs_on_device)

    # Store the result
    results[label] = OnnxInferenceResult(
        time_buffer,
        None
    )

PROVIDERS = {
    ("CPUExecutionProvider", "ONNX CPU"),
    #  Uncomment this line to enable GPU benchmarking
    #     ("CUDAExecutionProvider", "ONNX GPU")
}

# ONNX
for provider, label in PROVIDERS:
    # Create the model with the specified provider
    model = create_model_for_provider("onnx/bert-base-chinese.onnx", provider)

    # Keep track of the inference time
    time_buffer = []

    # Warm up the model
    model.run(None, inputs_onnx)

    # Compute
    for _ in trange(100, desc=f"Tracking inference time on {provider}"):
        with track_infer_time(time_buffer):
            model.run(None, inputs_onnx)

    # Store the result
    results[label] = OnnxInferenceResult(
        time_buffer,
        model.get_session_options().optimized_model_filepath
    )


# ONNX opt

PROVIDERS_OPT = {
    ("CPUExecutionProvider", "ONNX opt CPU")
}

for provider, label in PROVIDERS_OPT:
    # Create the model with the specified provider
    model = create_model_for_provider("onnx/bert-base-chinese.opt.onnx", provider)

    # Keep track of the inference time
    time_buffer = []

    # Warm up the model
    model.run(None, inputs_onnx)

    # Compute
    for _ in trange(100, desc=f"Tracking inference time on {provider}"):
        with track_infer_time(time_buffer):
            model.run(None, inputs_onnx)

    # Store the result
    results[label] = OnnxInferenceResult(
        time_buffer,
        model.get_session_options().optimized_model_filepath
    )

# 将 result save 处理, 绘制结果对比图
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)