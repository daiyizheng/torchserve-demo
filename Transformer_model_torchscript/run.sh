
echo "***********开始*************"
echo "path: "$(pwd)
echo "***********************"
## 打包
/usr/local/python3/bin/torch-model-archiver --model-name BERTSeqClassification1 --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler Transformer_model/Transformer_handler_generalized.py --extra-files "Transformer_model/setup_config.json,Transformer_model/index_to_name.json"
mkdir model_store
mv BERTSeqClassification1.mar model_store/
## 启动服务
/usr/local/python3/bin/torchserve --start --model-store model_store --ts-config Transformer_model/config.properties --models my_tc=BERTSeqClassification1.mar --ncs

echo "***********结束*************"