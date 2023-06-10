cmake_minimum_required(VERSION 3.0)

project(vrneat)

file(GLOB SOURCES src/*.cpp)

include_directories(./include)

add_library(vrneat STATIC ${SOURCES})

target_compile_options(vrneat PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror -O3 -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wno-error=maybe-uninitialized -Wpedantic -Wcast-align>
)

target_include_directories(vrneat PUBLIC include/)
