cmake_minimum_required(VERSION 3.10)

project(ttscore)
include_directories(./include)
add_library(ttscore SHARED ./src/ttscore.cpp)
target_link_libraries(ttscore python3.6m)