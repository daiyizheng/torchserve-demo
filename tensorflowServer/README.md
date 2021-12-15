
## 依赖
```bash
numpy==1.19.2
tensorflow-cpu==1.15.0
tensorflow-serving-api==1.15.0
```
## 服务已经起来，API接口未调通

## 单模型
```shell
docker run -p 8501:8501 -p 8500:8500 \
--mount type=bind,source=$(pwd)/multiModel/,target=/models/multiModel/ \
-e MODEL_NAME=bert -t tensorflow/serving
```
## 多模型
```shell
## cpu
docker run -p 8525:8525 -p 8526:8526 --name=tfserver \
--mount type=bind,source=$(pwd)/multiModel/,target=/models/multiModel/ \
-t --entrypoint=tensorflow_model_server tensorflow/serving \
--port=8525 --rest_api_port=8526 \
--enable_batching=true --file_system_poll_wait_seconds=300 \
--grpc_channel_arguments="grpc.max_connection_age_ms=5000" \
--model_config_file=/models/multiModel/models.config
```

## 多模型 models.config配置文件
```shell
model_config_list:{
    config:{
      name:"bert",
      base_path:"/models/multiModel/bert",
      model_platform:"tensorflow",
      model_version_policy:{
        all:{}
      }
    },
    config:{
      name:"bert2",
      base_path:"/models/multiModel/bert2",
      model_platform:"tensorflow",
      model_version_policy: {
       specific: {
        versions: 1
       }
    }
    },
    config:{
      name:"bert3",
      base_path:"/models/multiModel/bert3",
      model_platform:"tensorflow",
       model_version_policy: {
       latest: {
        num_versions: N
       }
    }
    }
}
```
请求预测的时候，如果要使用版本为100001的模型，只要修改SERVER_URL为：
`SERVER_URL = ‘http://localhost:8526/v1/models/bert/versions/1:predict'
tfserving支持模型的Hot Plug，上述容器运行起来之后，如果在宿主机的 $(pwd)/multiModel/bert/ 文件夹下新增模型文件如3/，tfserving会自动加载新模型；同样如果移除现有模型，tfserving也会自动卸载模型。

本示例来自nezha tensorflow版本
https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow

