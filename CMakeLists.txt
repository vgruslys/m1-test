cmake_minimum_required(VERSION 3.23)
project(metal_experiment)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_library(APPLE_FOUNDATION Foundation REQUIRED)
find_library(APPLE_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_METAL Metal REQUIRED)

message(${APPLE_FOUNDATION})
message(${APPLE_QUARTZ_CORE})
message(${APPLE_METAL})

add_executable(my_exe prog.cpp)

target_link_libraries(
    my_exe
    ${APPLE_FOUNDATION}
    ${APPLE_QUARTZ_CORE}
    ${APPLE_METAL}
)
