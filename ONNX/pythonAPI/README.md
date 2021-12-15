
经热心网友冠达提醒优化的ONNX模型运行时要开启OpenMP（如果没有安装，用apt-get install libgomp1安装OpenMP运行时库即可）
## 依赖
pip install onnxruntime       # CPU 版本
pip install onnxruntime-gpu   # GPU 版本
