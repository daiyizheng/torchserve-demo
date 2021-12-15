# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 17:50
# software: PyCharm

"""
文件说明：
    
"""
import netron
modelPath = "./onnx/bert-base-chinese.onnx"
netron.start(modelPath)