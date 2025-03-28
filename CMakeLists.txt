cmake_minimum_required(VERSION 3.14)
project(onnx2trt CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(TENSORRT_PATH "/home/lingxh/Downloads/TensorRT-8.6.1.6")
include_directories(${TENSORRT_PATH}/include)
file(GLOB LIBS "${TENSORRT_PATH}/lib/*.so")

find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/tools)

add_executable(onnx2trt onnx2trt.cpp)
add_executable(infer_trt infer_trt.cpp)

target_link_libraries(onnx2trt ${LIBS})
target_link_libraries(infer_trt ${LIBS} ${CUDA_cudart_LIBRARY})
