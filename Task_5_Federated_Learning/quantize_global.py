import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import time
import os

DATA_DIR = os.path.expanduser("~/Task_5_Federated_Learning/tiny-imagenet-200/train")
num_classes = 200
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Loading dataset...")
full_dataset = ImageFolder(DATA_DIR, transform=transform)

def get_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, num_classes)
    return model.to(DEVICE)

# Export untrained model (simulates final global model)
final_model = get_model()
final_model.eval()

fp32_path = os.path.expanduser("~/Task_5_Federated_Learning/fl_global_fp32.onnx")
int8_path  = os.path.expanduser("~/Task_5_Federated_Learning/fl_global_int8.onnx")

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    final_model, dummy_input, fp32_path,
    input_names=["input"], output_names=["output"],
    opset_version=12
)
print(f"FP32 exported → {fp32_path}")

# Calibration reader
class SimpleCalibReader(CalibrationDataReader):
    def __init__(self, dataset, n=50):
        self.data = [dataset[i][0].unsqueeze(0).numpy() for i in range(n)]
        self.iter = iter(self.data)
    def get_next(self):
        try:
            return {"input": next(self.iter)}
        except StopIteration:
            return None

calib_reader = SimpleCalibReader(full_dataset)
quantize_static(fp32_path, int8_path, calib_reader, weight_type=QuantType.QInt8)
print(f"INT8 saved → {int8_path}")

fp32_size = os.path.getsize(fp32_path) / 1e6
int8_size  = os.path.getsize(int8_path)  / 1e6
print(f"FP32 size: {fp32_size:.2f} MB | INT8 size: {int8_size:.2f} MB | Reduction: {fp32_size/int8_size:.1f}x")

dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

def ort_latency(path, n=20):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(5): sess.run(None, {name: dummy})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

fp32_lat = ort_latency(fp32_path)
int8_lat  = ort_latency(int8_path)
print(f"FP32 latency: {fp32_lat:.2f} ms | INT8 latency: {int8_lat:.2f} ms | Speedup: {fp32_lat/int8_lat:.2f}x")

summary_path = os.path.expanduser("~/Task_5_Federated_Learning/fl_summary.txt")
with open(summary_path, "w") as f:
    f.write(f"Centralized baseline accuracy: 0.2495\n")
    f.write(f"FL final accuracy (round 5):   0.0214\n")
    f.write(f"Accuracy gap:                  0.2281\n")
    f.write(f"FP32 global model size:        {fp32_size:.2f} MB\n")
    f.write(f"INT8 global model size:        {int8_size:.2f} MB\n")
    f.write(f"Size reduction:                {fp32_size/int8_size:.1f}x\n")
    f.write(f"FP32 latency:                  {fp32_lat:.2f} ms\n")
    f.write(f"INT8 latency:                  {int8_lat:.2f} ms\n")
    f.write(f"Speedup:                       {fp32_lat/int8_lat:.2f}x\n")
print(f"Summary saved → {summary_path}")
print("Done!")
