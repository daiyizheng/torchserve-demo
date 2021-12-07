# TorchServe gRPC API

## TorchServe 提供以下 gRPCs api

- [推理 API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto)
    - Ping : 获取正在运行的服务器的健康状态
    - Predictions：从提供的模型中获取预测

- [管理接口]()
    - RegisterModel：在 TorchServe 上提供模型/模型版本
    - UnregisterModel：通过从 TorchServe 取消注册模型的特定版本来释放系统资源
    - ScaleWorker：动态调整模型的任何版本的工作人员数量，以更好地服务于不同的推理请求负载。
    - ListModels : 查询当前注册模型的默认版本
    - DescribeModel：获取模型默认版本的详细运行时状态
    - SetDefault：将模型的任何注册版本设置为默认版本

默认情况下，TorchServe 在端口 7070 上侦听 gRPC Inference API，在 7071 上侦听 gRPC Management API。要在不同端口上配置 gRPC API，请参阅配置[文档](https://github.com/pytorch/serve/blob/master/docs/configuration.md)

## 依赖安装
torchserve相关
`pip install torchserve torch-model-archiver torch-workflow-archiver`
GRPC相关
`pip install -U grpcio protobuf grpcio-tools`

## 启动torchserve
```bash
mkdir model_store
torchserve --start --ts-config ./config.properties  --model-store model_store --ncs
```

## 使用 proto 文件生成 python gRPC客户端

```bash
python -m grpc_tools.protoc --proto_path=proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts proto/inference.proto proto/management.proto
```

## densenet161模型注册

```bash
python ts_scripts/torchserve_grpc_client.py register densenet161
```

## densenet161模型推理

```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

## densenet161模型注销
```bash
python ts_scripts/torchserve_grpc_client.py unregister densenet161
```
