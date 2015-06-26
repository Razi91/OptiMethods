cmake_minimum_required(VERSION 2.8)
project(metody)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(metody ${SOURCE_FILES} Function.cpp Function.h Fun1.cpp Fun1.h NR.h utils.h Matrix.cpp Matrix.h Vec.cpp Vec.h Fun2.cpp Fun2.h)