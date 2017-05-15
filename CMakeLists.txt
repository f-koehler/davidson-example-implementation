cmake_minimum_required(VERSION 2.8)
project(davidson LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories(eigen)

add_executable(davidson davidson.cpp)
