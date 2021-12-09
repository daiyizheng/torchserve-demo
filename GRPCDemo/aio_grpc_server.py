# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/8 15:41
# software: PyCharm

"""
文件说明：
    aio的服务端实现方式：
"""
import os
import sys
import time
import grpc
from grpc.experimental import aio
import asyncio
from concurrent import futures

from proto import helloword_pb2, helloword_pb2_grpc

class Greeter(helloword_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        print(request.name)
        message = f"This message is from Server.And what i want to say is hello { request.name }"
        return helloword_pb2.HelloReply(message = message)

async def start_server():
    # start rpc service
    server = aio.server(futures.ThreadPoolExecutor(max_workers=40), options=[
        ('grpc.so_reuseport', 0),
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ('grpc.enable_retries', 1),
    ])
    helloword_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)  # 加入服务

    server.add_insecure_port('[::]:50051')
    await server.start()

    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(None)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait([start_server()]))
    loop.close()