cmake_minimum_required(VERSION 3.16)

project(vegapilot_rt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    vegapilot_rt

    src/main.cpp
    src/pid.cpp
    src/mixer.cpp
    src/controller.cpp
)

target_include_directories(
    vegapilot_rt
    PRIVATE
    include
)