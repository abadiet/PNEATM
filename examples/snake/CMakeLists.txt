cmake_minimum_required(VERSION 3.0)

project(SnakePNEATM)

include_directories(../../include)
add_library(pneatm STATIC IMPORTED)
set_property(TARGET pneatm PROPERTY IMPORTED_LOCATION ../../../lib/PNEATM/libpneatm.a)

add_executable(SnakePNEATM main.cpp)

file(GLOB SOURCES src/*.cpp)
target_sources(SnakePNEATM PRIVATE ${SOURCES})
include_directories(include)

target_compile_options(SnakePNEATM PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror -O3 -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wno-error=maybe-uninitialized -Wpedantic>
)

target_link_libraries(SnakePNEATM pneatm sfml-window sfml-system sfml-graphics)
