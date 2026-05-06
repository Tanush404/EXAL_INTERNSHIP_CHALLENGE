# Task 3 — C++ Inference Wrapper

## Overview
A C++ binary that loads an ONNX model and runs inference using the ONNX Runtime C++ API.

## Requirements
- CMake >= 3.16
- g++ with C++17 support
- ONNX Runtime v1.20.1 (Linux x64)

## Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```

## Run
```bash
./build/infer --model mobilenet_v2.onnx --input ./images --batch 1 --threads 2
```

## Memory Check
```bash
valgrind --leak-check=full ./build/infer --model mobilenet_v2.onnx --input ./images
```

## Results
- Inference latency: ~4-11ms per image on CPU
- Memory: 192 bytes lost from ONNX Runtime internal thread pool (known upstream issue)
- No leaks in user code

