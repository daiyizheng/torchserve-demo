# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/14 17:33
# software: PyCharm

"""
文件说明：
    
"""
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('onnx/bert-base-chinese.opt.onnx',None)
inp0 = np.array([[101, 1920, 2157, 1962,  117, 2769, 3221, 1297, 1147, 5130, 4638, 2207,
                  4511, 2111,  117, 3684,  689,  754, 1290,  704, 4906, 2825, 1920, 2110,
                  102]], dtype=np.int64)

inp1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0]], dtype=np.int64)

inp2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0]], dtype=np.int64)

scores = sess.run(['output_0', 'output_1'],{'input_ids': inp0,'attention_mask':inp1,'token_type_ids':inp2})
## 或者 scores = sess.run(None,{'input_ids': inp0,'attention_mask':inp1,'token_type_ids':inp2})
print('模型结果数据维度：',scores[0].shape)