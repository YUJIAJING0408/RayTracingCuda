cmake_minimum_required(VERSION 3.31)
project(RaytracingCPU CXX)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
set(EXECUTABLE_OUTPUT_PATH  ${BIN_PATH})
add_executable(RaytracingCPU main.cpp)

