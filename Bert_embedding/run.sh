
echo "***********开始*************"
echo "path: "$(pwd)
echo "***********************"
## 打包
/usr/local/python3/bin/torch-model-archiver --model-name bertEmbbeding --version 1.0 --serialized-file bert_embbeding/traced_model.pt --handler bert_embbeding/BERTHandler.py --extra-files "bert_embbeding/setup_config.json"
mkdir model_store
mv bertEmbbeding.mar model_store/
## 启动服务
/usr/local/python3/bin/torchserve --start --model-store model_store --ts-config bert_embbeding/config.properties --models bertEmbbeding=bertEmbbeding.mar --ncs

echo "***********结束*************"