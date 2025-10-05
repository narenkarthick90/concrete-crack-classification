import onnxruntime as ort
import numpy as np
import time
import psutil, os

onnx_model = "crack_classifier.onnx"
session = ort.InferenceSession(onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Setting up
for _ in range(10):
    _ = session.run(None, {input_name: dummy})

# Actual Measuring
start = time.time()
for _ in range(100):
    _ = session.run(None, {input_name: dummy})
end = time.time()

avg_time = (end - start) / 100
fps = 1 / avg_time
mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

print(f"Average inference time: {avg_time*1000:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Memory usage: {mem:.2f} MB")
