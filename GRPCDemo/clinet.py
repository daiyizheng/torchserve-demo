# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/8 15:16
# software: PyCharm

"""
文件说明：
    
"""
#coding=utf-8
# from __future__ import print_function

import grpc

from proto import helloword_pb2, helloword_pb2_grpc



def run():
    channel = grpc.insecure_channel('localhost:50051') # 案例采用未认证的insecure_channel方式，生产环境建议使用secure_channel方式。
    stub = helloword_pb2_grpc.GreeterStub(channel)
    response = stub.SayHello(helloword_pb2.HelloRequest(name='Hello World！ This is message from client!'))
    print("Greeter client received: " + response.message)


if __name__ == '__main__':
    run()