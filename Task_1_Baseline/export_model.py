import sys
import os
sys.path.insert(0, 'C:\\pylib')

import torch
import torchvision

# 1. Load MobileNetV2 pretrained on ImageNet
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval() 

# 2. Create 'dummy' input (a fake 224x224 image)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX format
torch.onnx.export(model, dummy_input, "mobilenet_v2.onnx", 
                  export_params=True, 
                  opset_version=12)

print("Success: 'mobilenet_v2.onnx' created in your current folder.")
