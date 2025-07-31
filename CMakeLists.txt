cmake_minimum_required(VERSION 3.31)
project(Raytracing CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(BIN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_subdirectory(src/cpu)
add_subdirectory(src/cuda)