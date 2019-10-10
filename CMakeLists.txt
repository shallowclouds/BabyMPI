CMAKE_MINIMUM_REQUIRED(VERSION 3.4)

set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)
add_executable(source source.cpp)
add_executable(optimize1 optimize1.cpp)
add_executable(optimize2 optimize2.cpp)
add_executable(optimize3 optimize3.cpp)
