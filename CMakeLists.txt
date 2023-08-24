cmake_minimum_required(VERSION 3.5)

# Options
option(BUILD_EXAMPLES "Build Examples" ON)

# Project
project(pneatm)

# Sources
file(GLOB_RECURSE SOURCES src/*.cpp)

# Add the library
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Includes
target_include_directories(pneatm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Flags
target_compile_options(${PROJECT_NAME} PRIVATE
	$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
	$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra -Werror -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wno-error=maybe-uninitialized -Wpedantic>
)

# Add Libraries
find_package(spdlog REQUIRED)
find_package(SFML COMPONENTS window system graphics REQUIRED)

# Link Libraries
target_link_libraries(${PROJECT_NAME}
	PUBLIC spdlog::spdlog
	PRIVATE sfml-window sfml-system sfml-graphics
)

if(BUILD_EXAMPLES)
	# SnakePNEATM
    add_subdirectory(examples/snake)
endif()