# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/16 13:07
# software: PyCharm

"""
文件说明：
    
"""
from txtai.embeddings import Embeddings
from txtai.pipeline import Labels

labels = Labels(("text-classify.onnx", "google/electra-base-discriminator"), dynamic=False)
print(labels(["I am happy", "I am mad"]))

embeddings = Embeddings({"path": "embeddings.onnx", "tokenizer": "sentence-transformers/paraphrase-MiniLM-L6-v2"})
print(embeddings.similarity("I am happy", ["I am glad"]))
