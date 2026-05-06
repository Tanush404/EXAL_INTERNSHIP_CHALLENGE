cmake_minimum_required(VERSION 3.16)
project(infer)

set(CMAKE_CXX_STANDARD 17)
# Path to ONNX Runtime
set(ORT_ROOT "$ENV{HOME}/onnxruntime-linux-x64-1.20.1")

include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

add_executable(infer src/infer.cpp)

target_link_libraries(infer
    ${ORT_ROOT}/lib/libonnxruntime.so
)
