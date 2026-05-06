#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

// Resize and normalize a raw image into a float tensor (224x224x3 -> 1x3x224x224)
// We simulate preprocessing: fill with dummy normalized values
// In real use, you'd decode JPEG here using stb_image or OpenCV
std::vector<float> preprocessImage(const std::string& imagePath) {
    // 1x3x224x224 = 150528 floats
    std::vector<float> tensor(1 * 3 * 224 * 224);

    // ImageNet normalization constants
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_dev[3] = {0.229f, 0.224f, 0.225f};

    // Fill with normalized mid-grey (placeholder for real JPEG decode)
    for (int c = 0; c < 3; c++) {
        float val = (0.5f - mean[c]) / std_dev[c];
        for (int i = 0; i < 224 * 224; i++) {
            tensor[c * 224 * 224 + i] = val;
        }
    }
    return tensor;
}

int main(int argc, char* argv[]) {

    // --- Parse CLI arguments ---
    std::string modelPath = "model.onnx";
    std::string inputDir  = "./images";
    int batchSize         = 1;
    int numThreads        = 2;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model")   == 0 && i+1 < argc) modelPath  = argv[++i];
        if (strcmp(argv[i], "--input")   == 0 && i+1 < argc) inputDir   = argv[++i];
        if (strcmp(argv[i], "--batch")   == 0 && i+1 < argc) batchSize  = std::stoi(argv[++i]);
        if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) numThreads = std::stoi(argv[++i]);
    }

    std::cout << "Model:   " << modelPath  << "\n";
    std::cout << "Input:   " << inputDir   << "\n";
    std::cout << "Batch:   " << batchSize  << "\n";
    std::cout << "Threads: " << numThreads << "\n\n";

    // --- Set up ONNX Runtime ---
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "infer");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(numThreads);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names
    auto inputNamePtr  = session.GetInputNameAllocated(0, allocator);
    auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
    std::string inputName  = inputNamePtr.get();
    std::string outputName = outputNamePtr.get();

    std::cout << "Input node:  " << inputName  << "\n";
    std::cout << "Output node: " << outputName << "\n\n";

    // Input shape: [1, 3, 224, 224]
    std::array<int64_t, 4> inputShape = {1, 3, 224, 224};
    size_t inputSize = 1 * 3 * 224 * 224;

    // --- Loop over images ---
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg") {
            imageFiles.push_back(entry.path().string());
        }
    }

    if (imageFiles.empty()) {
        std::cerr << "No .jpg images found in " << inputDir << "\n";
        return 1;
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    std::cout << "Found " << imageFiles.size() << " image(s)\n\n";
    std::cout << "Filename | Predicted Class | Confidence | Latency (ms)\n";
    std::cout << std::string(65, '-') << "\n";

    for (const auto& imgPath : imageFiles) {

        // Preprocess
        std::vector<float> inputTensor = preprocessImage(imgPath);

        // Create input tensor
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensor.data(),
            inputSize,
            inputShape.data(),
            inputShape.size()
        );

        // Run inference and measure time
        const char* inputNames[]  = {inputName.c_str()};
        const char* outputNames[] = {outputName.c_str()};

        auto t0 = std::chrono::high_resolution_clock::now();

        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,  &inputOrtTensor, 1,
            outputNames, 1
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double latencyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Get output (1000 class scores for ImageNet)
        float* scores = outputTensors[0].GetTensorMutableData<float>();
        int64_t numClasses = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape()[1];

        // Find top class
        int bestClass = 0;
        float bestScore = scores[0];
        for (int i = 1; i < numClasses; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestClass = i;
            }
        }

        // Softmax for confidence
        float expSum = 0.0f;
        std::vector<float> probs(numClasses);
        for (int i = 0; i < numClasses; i++) {
            probs[i] = std::exp(scores[i] - bestScore);
            expSum += probs[i];
        }
        float confidence = probs[bestClass] / expSum;

        // Print result
        std::string fname = fs::path(imgPath).filename().string();
        std::cout << fname << " | class " << bestClass
                  << " | " << confidence * 100.0f << "%"
                  << " | " << latencyMs << " ms\n";
    }

    return 0;
}

