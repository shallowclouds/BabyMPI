CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
add_executable(source source.cpp)
