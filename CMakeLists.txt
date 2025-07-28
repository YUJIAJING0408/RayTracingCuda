cmake_minimum_required(VERSION 3.31)
project(RaytracingCuda  CUDA)
project(ppmTest CXX)

set(CMAKE_CUDA_STANDARD 26)
set(CMAKE_CUDA_COMPILER "C:/cuda/bin/nvcc.exe")  # 根据您的CUDA版本调整路径

set(HEADER_DIR "C:/cuda/include")
set(LIB_DIR "C:/cuda/lib/x64")

include_directories(${HEADER_DIR})
link_directories(${LIB_DIR})
link_libraries(cufft nppial nppist nvml cufftw nppicc nppisu nvrtc curand nppitc OpenCL cuda cusolver nppidei npps cudadevrt cusparse nppif nvblas cudart nppc nppig cudart_static nppim)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
add_executable(RaytracingCuda main.cu)
add_executable(ppmTest src/utils/ppm.cpp)


set_target_properties(RaytracingCuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
