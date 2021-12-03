#! /bin/sh

## 打包
/usr/local/python3/bin/torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file ./Transformer_model_torchscript/traced_model.pt --handler ./Transformer_model_torchscript/Transformer_handler_generalized.py --extra-files "./Transformer_model_torchscript/setup_config.json,./Transformer_model_torchscript/index_to_name.json"

## 启动服务
/usr/local/python3/bin/torchserve --start --model-store model_store --ts-config model_store/config.properties --models my_tc=BERTSeqClassification.mar --ncs
