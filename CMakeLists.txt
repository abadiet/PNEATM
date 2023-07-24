cmake_minimum_required(VERSION 3.5)

# Project
project(pneatm)

# Sources
file(GLOB_RECURSE SOURCES src/*.cpp)

# Includes
include_directories(include)

# Add the library
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Flags
target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror -O3 -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wno-error=maybe-uninitialized -Wpedantic>
)