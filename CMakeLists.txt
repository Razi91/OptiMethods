cmake_minimum_required(VERSION 3.1)
project(metody)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES main.cpp)
add_executable(metody ${SOURCE_FILES} Function.cpp Function.h Fun1.cpp Fun1.h NR.cpp NR.h utils.h Matrix.cpp Matrix.h Vec.cpp Vec.h)