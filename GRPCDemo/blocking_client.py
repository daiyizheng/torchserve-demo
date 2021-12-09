# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/8 16:03
# software: PyCharm

"""
文件说明：
    
"""

import grpc
import sys
import os
from proto import helloword_pb2, helloword_pb2_grpc
from rpc_config import RPC_OPTIONS


class RpcClient(object):

    rpc_client = {}

    @staticmethod
    def get_rpc_channel(host, port, is_aio=False):
        options = RPC_OPTIONS
        # OPTIONS配置可根据需要自行设置：
        #RPC_OPTIONS = [('grpc.max_send_message_length', 100 * 1024 * 1024),
        #       ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        #       ('grpc.enable_retries', 1),
        #       ('grpc.service_config',
        #        '{"retryPolicy": {"maxAttempts": 4, "initialBackoff": "0.1s", '
        #        '"maxBackoff": "1s", "backoffMutiplier": 2, '
        #        '"retryableStatusCodes": ["UNAVAILABLE"]}}'),
        #       ]
        if is_aio:
            channel = grpc.aio.insecure_channel("{}:{}".format(host, port), options=options)
        else:
            channel = grpc.insecure_channel("{}:{}".format(host, port), options=options)
        return channel

    def load_sub_rpc(self, host, port, platform=None, db_type=None, is_aio=False):
        """
        function return rpc instance
        :param platform
        :param host
        :param port
        :param db_type
        :param is_aio
        :return: instance
        """
        channel = self.get_rpc_channel(host, port, is_aio)

        stub = helloword_pb2_grpc.GreeterStub(channel)
        return stub

def run():
    stub = RpcClient().load_sub_rpc('localhost',50051)
    response = stub.SayHello(helloword_pb2.HelloRequest(name='Hello World！ This is message from client!'))
    print("Greeter client received: " + response.message)


if __name__ == '__main__':
    run()