import os
import sys
import json
import numpy as np
from PIL import Image
sys.path.insert(0, 'C:\\pylib') 
import onnxruntime as ort


# This function maps Tiny ImageNet IDs (nxxxx) to the Model's 1000 original indices
def load_val_labels(anno_path, wnids_path):
    # 1. Get the list of 200 IDs used in Tiny ImageNet in order
    with open(wnids_path, 'r') as f:
        tiny_wnids = [x.strip() for x in f.readlines()]
    
    # 2. Map 'nxxxx' ID to a simple 0-199 index
    # Pretrained models often need this mapping to evaluate correctly
    wnid_to_idx = {wnid: i for i, wnid in enumerate(tiny_wnids)}
    
    val_labels = {}
    with open(anno_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t') # Tab Separation (EXAL Requirement)
            img_name = parts[0]
            wnid = parts[1]
            if wnid in wnid_to_idx:
                val_labels[img_name] = wnid_to_idx[wnid]
    return val_labels

# 2. Evaluation Logic
def evaluate(model_path, val_labels, img_dir):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Check if this is the FP16 model
    is_fp16 = "fp16" in model_path.lower()
    correct = 0
    test_set = list(val_labels.items())[:500]
    
    for filename, true_label_idx in test_set:
        img_path = os.path.join(img_dir, filename)
        img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
        
        # Standard Preprocessing
        img_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_data = np.expand_dims(img_data.transpose(2, 0, 1), axis=0)
        
        # FIX: If model is FP16, the input data MUST be float16
        if is_fp16:
            img_data = img_data.astype(np.float16)
        else:
            img_data = img_data.astype(np.float32)
        
        preds = session.run(None, {input_name: img_data})[0]
        if np.argmax(preds) == true_label_idx:
            correct += 1
            
    return (correct / len(test_set)) * 100

# --- Paths ---
root = r'C:\Users\durga\tiny-imagenet-200\tiny-imagenet-200'
anno_file = os.path.join(root, 'val', 'val_annotations.txt')
wnids_file = os.path.join(root, 'wnids.txt')
img_dir = os.path.join(root, 'val', 'images')

# Load Labels
labels = load_val_labels(anno_file, wnids_file)

# --- Execution ---
print(f"Testing FP32 Baseline...")
acc_fp32 = evaluate("mobilenet_v2.onnx", labels, img_dir)

print(f"Testing FP16 Variant...")
acc_fp16 = evaluate("mobilenet_v2_fp16.onnx", labels, img_dir)

print(f"Testing INT8 Optimized...")
acc_int8 = evaluate("mobilenet_v2_static.onnx", labels, img_dir)

# --- D2 RESULTS ---
print(f"\n--- D2 FINAL COMPARISON ---")
print(f"FP32 Accuracy: {acc_fp32:.2f}%")
print(f"FP16 Accuracy: {acc_fp16:.2f}%")
print(f"INT8 Accuracy: {acc_int8:.2f}%")

print(f"\n--- Accuracy Delta (Goal: < 2%) ---")
print(f"FP32 vs INT8: {acc_fp32 - acc_int8:.2f}%")