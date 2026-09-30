cmake_minimum_required(VERSION 3.16)
project(cuckoo_pa1 CXX)

# C++17 for aligned operator new.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(cuckoo main.cpp)

if(MSVC)
  target_compile_options(cuckoo PRIVATE /W4 /permissive-)
else()
  target_compile_options(cuckoo PRIVATE -Wall -Wextra -Wpedantic)
endif()
