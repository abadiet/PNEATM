cmake_minimum_required(VERSION 3.5)

if (PURE_CPP)
	message(SEND_ERROR "PURE_CPP is not supported for the snake example")
else()

	# Project
	project(SnakePNEATM)

	# Sources
	file(GLOB_RECURSE SOURCES src/*.cpp)
	file(GLOB_RECURSE HEADERS include/*.hpp)

	# Executable
	add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

	# Includes
	target_include_directories(${PROJECT_NAME} PRIVATE include)

	# Flags
	target_compile_options(${PROJECT_NAME} PRIVATE
		$<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
		$<$<CXX_COMPILER_ID:AppleClang>:-std=c++11>
		$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -Wall -Wextra -Werror -Wundef -Wcast-align -Wwrite-strings -Wunreachable-code -Wconversion -Wpedantic>
	)

	# Link Libraries
	target_link_libraries(${PROJECT_NAME} PRIVATE pneatm)

endif()
