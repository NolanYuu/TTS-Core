cmake_minimum_required(VERSION 3.10)

project(ttscore_api)
include_directories(./include)
include_directories(/usr/include/python3.8/)
add_library(ttscore_api SHARED ./src/ttscore_api.cpp)
target_link_libraries(ttscore_api python3.8)