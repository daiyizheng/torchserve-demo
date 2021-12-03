FROM registry.cn-hangzhou.aliyuncs.com/daiyizheng/centos-ssh-py37-java11:v2.0.0
## 工作目录
WORKDIR /root/model_deploy/Transformer_model_torchscript

## 复制项目
COPY ./  ./

## 安装除torchserver torch和transformer的依赖
RUN pip install --upgrade pip && pip install -i  https://mirrors.aliyun.com/pypi/simple -r requirements.txt

RUN chmod 777 run.sh
EXPOSE 8080/tcp
EXPOSE 8081/tcp

CMD ./run.sh


