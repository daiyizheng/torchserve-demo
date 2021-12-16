# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 11:58
# software: PyCharm

"""
文件说明：
    
"""
from datasets import load_dataset
from txtai.pipeline import HFTrainer
from txtai.pipeline import HFOnnx
from transformers import AutoTokenizer
import numpy as np

from io import BytesIO
from itertools import chain
from torch.onnx import export

# Conditional import
try:
    from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
    from onnxruntime.quantization import quantize_dynamic

    ONNX_RUNTIME = True
except ImportError:
    ONNX_RUNTIME = False

class MyHFOnnx(HFOnnx):
    def __call__(self, path, task="default", output=None, quantize=False, opset=12):
        """
        Exports a Hugging Face Transformer model to ONNX.

        Args:
            path: path to model, accepts Hugging Face model hub id, local path or (model, tokenizer) tuple
            task: optional model task or category, determines the model type and outputs, defaults to export hidden state
            output: optional output model path, defaults to return byte array if None
            quantize: if model should be quantized (requires onnx to be installed), defaults to False
            opset: onnx opset, defaults to 12

        Returns:
            path to model output or model as bytes depending on output parameter
        """

        inputs, outputs, model = self.parameters(task)

        if isinstance(path, (list, tuple)):
            model, tokenizer = path
            model = model.cpu()
        else:
            model = model(path)
            tokenizer = AutoTokenizer.from_pretrained(path)

        # Generate dummy inputs
        dummy = dict(tokenizer(["test inputs"], return_tensors="pt"))

        # Default to BytesIO if no output file provided
        output = output if output else BytesIO()
        args = [dummy[key] for key in list(inputs.keys())]

        # Export model to ONNX
        export(
            model,
            tuple(args),
            output,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(inputs.keys()),
            output_names=list(outputs.keys()),
            dynamic_axes=dict(chain(inputs.items(), outputs.items())),
        )

        # Quantize model
        if quantize:
            if not ONNX_RUNTIME:
                raise ImportError('onnxruntime is not available - install "pipeline" extra to enable')

            output = self.quantization(output)

        if isinstance(output, BytesIO):
            # Reset stream and return bytes
            output.seek(0)
            output = output.read()

        return output
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

onnx = MyHFOnnx()
trainer = HFTrainer()

# Hugging Face dataset
ds = load_dataset("glue", "sst2")
data = ds["train"].select(range(5000)).flatten_indices()

# Train new model using 5,000 SST2 records (in-memory)
model, tokenizer = trainer("google/electra-base-discriminator", data, columns=("sentence", "label"))

text = onnx((model, tokenizer), "text-classification", "text-classify.onnx", quantize=True)
embeddings = onnx("sentence-transformers/paraphrase-MiniLM-L6-v2", "pooling", "embeddings.onnx", quantize=True)