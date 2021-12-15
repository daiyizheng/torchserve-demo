# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 17:29
# software: PyCharm

"""
文件说明：
    
"""
from pathlib import Path
from transformers.convert_graph_to_onnx import convert

# Handles all the above steps for you
convert(framework="pt",
        model="bert-base-chinese",
        output=Path("onnx/bert-base-chinese.onnx"),
        opset=11)