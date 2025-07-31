cmake_minimum_required(VERSION 3.31)
project(RaytracingCuda CXX CUDA)

set(CMAKE_CUDA_STANDARD 26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe")  # 根据您的CUDA版本调整路径
set(HEADER_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include")
set(LIB_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64")

include_directories(${HEADER_DIR})
include_directories(includes)
link_directories(${LIB_DIR})
link_libraries(cufft nppial nppist nvml cufftw nppicc nppisu nvrtc curand nppitc OpenCL cuda cusolver nppidei npps cudadevrt cusparse nppif nvblas cudart nppc nppig cudart_static nppim)
set(EXECUTABLE_OUTPUT_PATH  ${BIN_PATH})
add_executable(RaytracingCuda main.cu)


set_target_properties(RaytracingCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
