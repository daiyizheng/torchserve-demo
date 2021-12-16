# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 9:20
# software: PyCharm

"""
文件说明：
    
"""
import onnx

# Load the ONNX model
model = onnx.load("./onnx/bert-base-chinese.onnx")

# Check that the model is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))