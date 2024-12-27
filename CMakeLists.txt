cmake_minimum_required(VERSION 3.5)

# Options
cmake_policy(SET CMP0077 NEW)
option(BUILD_EXAMPLES "Build Examples" ON)
option(PURE_CPP "Pure C++" OFF)

# Project
project(pneatm)

# Sources
file(GLOB_RECURSE SOURCES src/*.cpp)

# Add the library
add_library(${PROJECT_NAME} STATIC ${SOURCES})

# Includes
target_include_directories(pneatm PUBLIC include)

# Flags
if (PURE_CPP)
	target_compile_options(${PROJECT_NAME} PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
		$<$<CXX_COMPILER_ID:AppleClang>:-std=c++11>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-DPURE_CPP -O3 -Wall -Wextra -Werror -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wpedantic>
	)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
		$<$<CXX_COMPILER_ID:AppleClang>:-std=c++11>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra -Werror -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wpedantic>
	)
endif()

if (NOT PURE_CPP)
	message(STATUS "Building with additional libraries (spdlog, SFML)")

	# Add Libraries
	find_package(spdlog REQUIRED)
	find_package(SFML COMPONENTS window system graphics REQUIRED)

	# Link Libraries
	target_link_libraries(${PROJECT_NAME}
		PUBLIC spdlog::spdlog
		PRIVATE sfml-window sfml-system sfml-graphics
	)
endif()

if(BUILD_EXAMPLES)
	# SnakePNEATM
    add_subdirectory(examples/snake)
endif()