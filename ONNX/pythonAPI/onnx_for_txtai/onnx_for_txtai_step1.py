# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 10:03
# software: PyCharm

"""
文件说明：
    
"""
## https://dev.to/neuml/export-and-run-models-with-onnx-fof

# STEP 1
## pip install txtai[pipeline]

import numpy as np

from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer
from txtai.pipeline import HFOnnx

from collections import OrderedDict
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

# Normalize logits using sigmoid function
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# Export to ONNX
onnx = MyHFOnnx()
model = onnx("bert-base-uncased", "text-classification", opset=11)

# Start inference session
options = SessionOptions()
session = InferenceSession(model, options)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(["I am happy", "I am mad"], return_tensors="np")

# Print results
outputs = session.run(None, dict(tokens))
print(sigmoid(outputs[0]))