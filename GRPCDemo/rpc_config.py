# -*- coding:UTF-8 -*-

# author:user
# contact: test@test.com
# datetime:2021/12/8 16:07
# software: PyCharm

"""
文件说明：
    
"""
RPC_OPTIONS = [('grpc.max_send_message_length', 100 * 1024 * 1024),
      ('grpc.max_receive_message_length', 100 * 1024 * 1024),
      ('grpc.enable_retries', 1),
      ('grpc.service_config',
       '{"retryPolicy": {"maxAttempts": 4, "initialBackoff": "0.1s", '
       '"maxBackoff": "1s", "backoffMutiplier": 2, '
       '"retryableStatusCodes": ["UNAVAILABLE"]}}'),
      ]
