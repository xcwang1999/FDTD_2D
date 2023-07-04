cmake_minimum_required(VERSION 3.25)
project(FDTD_2D CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
add_executable(executable FDTD2D.cu)
set_target_properties(executable PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)