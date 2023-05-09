cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)
project (kernel VERSION 0.1 DESCRIPTION "initial implementation")

add_library(kernel SHARED kernel.cpp)

set_target_properties(kernel PROPERTIES
  VERSION ${PROJECT_VERSION})
