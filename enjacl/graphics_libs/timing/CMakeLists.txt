
project(timing)
cmake_minimum_required(VERSION 2.6)
set(timings timingGE.cpp)


include_directories( . )
add_library(timer STATIC ${timings})
