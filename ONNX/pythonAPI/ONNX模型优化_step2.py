# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/15 10:30
# software: PyCharm

"""
文件说明：
通过使用特定的后端来进行inference，后端将启动特定硬件的graph优化。有3种基本的优化：

- Constant Folding: 将graph中的静态变量转换为常量
- Deadcode Elimination: 去除graph中未使用的nodes
- Operator Fusing: 将多条指令合并为一条(比如，Linear -> ReLU 可以合并为 LinearReLU)
在ONNX Runtime中通过设置特定的SessionOptions会自动使用大多数优化。
注意:一些尚未集成到ONNX Runtime 中的最新优化可在优化脚本中找到，利用这些脚本可以对模型进行优化以获得最佳性能。
    
# """


from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
opt_options = BertOptimizationOptions('bert')
opt_options.enable_embed_layer_norm = False
optimized_model = optimizer.optimize_model("onnx/bert-base-chinese.onnx",
                                           model_type='gpt2',
                                           num_heads=12,
                                           hidden_size=768,
                                           optimization_options=opt_options)
optimized_model.convert_float_to_float16()
optimized_model.save_model_to_file("onnx/bert-base-chinese.opt.onnx")

