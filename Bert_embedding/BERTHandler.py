# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/6 9:40
# software: PyCharm

"""
文件说明：
    
"""
from abc import ABC
import os
import json
import logging
import ast

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from ts.torch_handler.base_handler import BaseHandler


from transformers import BertModel, AutoTokenizer

logger = logging.getLogger(__name__)

class BERTHandler(BaseHandler, ABC):
    def __init__(self):
        super(BERTHandler, self).__init__()
        self.initialized = False
    def initialize(self, ctx):
        """模型初始化"""
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        ## 模型根目录
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"] # 对应 --serialized-file的参数
        model_pt_path = os.path.join(model_dir, serialized_file)

        ## setup_config的配置文件
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None and self.setup_config['is_gpu']
            else "cpu"
        )
        ## 加载模型
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path, map_location=self.device)
        elif self.setup_config["save_mode"] == "pretrained":
            self.model = BertModel.from_pretrained(model_dir)

        ## 加载tokenizer分词器
        if any(fname for fname in os.listdir(model_dir) if fname.startswith("vocab.") and os.path.isfile(fname)):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, do_lower_case=self.setup_config["do_lower_case"]
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.setup_config["model_name"],
                do_lower_case=self.setup_config["do_lower_case"],
            )

        self.model.to(self.device)
        self.model.eval()
        logger.info(
            "Transformer model from path %s loaded successfully", model_dir
        )

        ## 加载标签映射文件
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        if os.path.isfile(mapping_file_path):
            with open(mapping_file_path) as f:
                self.mapping = json.load(f)
        else:
            logger.warning("Missing the index_to_name.json file.")

        self.initialized = True

    def preprocess(self, requests):
        """基本文本预处理
        [{'body': {'text': ['Bloomberg']}}]
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode('utf-8')
            input_text_target = ast.literal_eval(input_text)
            input_text = input_text_target[self.setup_config["columns"]]
            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)
            inputs = self.tokenizer.batch_encode_plus(input_text,
                                                max_length=int(max_length),
                                                pad_to_max_length=True,
                                                add_special_tokens=True,
                                                return_tensors='pt')
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat((attention_mask_batch, attention_mask), 0)

        dataset = TensorDataset(input_ids_batch, attention_mask_batch)
        return dataset



    def inference(self, dataset):
        """
        模型推理
        """
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=int(self.setup_config["batch_size"]))

        # Eval!
        logger.info("***** Running evaluation on dev dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", int(self.setup_config["batch_size"]))

        self.model.eval()

        inferences = None
        for batch in tqdm(eval_dataloader, desc="Prediction"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1]}
                outputs = self.model(**inputs)  # (sequence_output, pooler_output)
                if inferences is None:
                    inferences = outputs[1].detach().cpu().numpy()
                else:
                    inferences = np.append(inferences, outputs[1].detach().cpu().numpy(), axis=0)
        return inferences.tolist()


    def postprocess(self, inference_output):
        """模型推理后处理 返回必须时数[[]]组"""

        return [inference_output]




