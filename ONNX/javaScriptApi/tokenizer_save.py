# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 14:16
# software: PyCharm

"""
文件说明：
    
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
tokenizer.save_pretrained("bert")