FROM registry.cn-hangzhou.aliyuncs.com/daiyizheng/centos-ssh-py37-java11:v2.0.0
## 工作目录
WORKDIR /root/model_deploy

RUN mkdir bert_embbeding

## 复制项目
COPY ./  ./bert_embbeding

## 安装除torchserver torch和transformer的依赖
RUN pip install --upgrade pip && pip install -i  https://mirrors.aliyun.com/pypi/simple -r bert_embbeding/requirements.txt

EXPOSE 8080/tcp
EXPOSE 8081/tcp

RUN chmod 777 ./bert_embbeding/run.sh
#
#
CMD ["bash","-c","./bert_embbeding/run.sh start && tail -f /dev/null"]


