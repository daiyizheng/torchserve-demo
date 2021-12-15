


## 模型挂载
容器模型路径：`/opt/ml/model`

```shell
dos2unix run.sh
docker build -t repu/torchserve:v1.0.0 .
docker run -dit  -p 8080:8080 -p 8081:8081 -v ${pwd}:/opt/ml/model --name bert  repu/torchserve:v1.0.0
```

## 注册模型
