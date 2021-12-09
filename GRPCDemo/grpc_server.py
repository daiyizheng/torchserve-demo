# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/8 13:10
# software: PyCharm

"""
文件说明：
    
"""
import os
import sys
import time
import grpc
import asyncio
from concurrent import futures
from proto import helloword_pb2, helloword_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class Greeter(helloword_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        print(request.name)
        message = f"This message is from Server.And what i want to say is hello { request.name }"
        return helloword_pb2.HelloReply(message = message)

options = [
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ('grpc.enable_retries', 1),
]
async def start_server():
    # start rpc service
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=40), options=options)
    helloword_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)  # 加入服务

    server.add_insecure_port('[::]:50051')
    server.start()
    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([start_server()]))
    loop.close()
