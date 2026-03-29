import os    
import sys
sys.path.insert(0, 'C:\\pylib') 

import onnxruntime as ort
import numpy as np
import time

# --- THIS IS THE SECTION FOR SIZE ---
# It looks for the files you created in Phase 1
onnx_size = os.path.getsize("mobilenet_v2.onnx")
data_size = os.path.getsize("mobilenet_v2.onnx.data")
total_size_mb = (onnx_size + data_size) / (1024 * 1024)
# ------------------------------------

# Load the model
session = ort.InferenceSession("mobilenet_v2.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
test_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Warm-up (The CPU needs to "wake up" before we measure)
for _ in range(10):
    session.run(None, {input_name: test_data})

# Measuring Speed (Latency)
latencies = []
for _ in range(100):
    start_time = time.perf_counter()
    session.run(None, {input_name: test_data})
    latencies.append((time.perf_counter() - start_time) * 1000)

# Printing the Results
print(f"--- Week 1 Baseline Results ---")
print(f"Total Model Size: {total_size_mb:.2f} MB") # This uses the math from above
print(f"P50 Latency: {np.percentile(latencies, 50):.2f} ms")
print(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
print(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
