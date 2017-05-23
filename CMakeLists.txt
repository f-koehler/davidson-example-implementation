cmake_minimum_required(VERSION 2.8)
project(davidson LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model precise")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -m64")
endif()

include_directories(eigen)

link_directories($ENV{MKLROOT}/lib/intel64)
include_directories($ENV{MKLROOT}/include)

add_executable(lowest lowest.cpp)
target_link_libraries(lowest mkl_intel_lp64 mkl_intel_thread mkl_core iomp5 pthread m dl)
