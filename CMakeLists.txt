cmake_minimum_required(VERSION 3.15)
project(CED_Schedule LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
    src/Multimethod.cpp
    src/Problems.cpp
    src/CC_HIHH.cpp
    src/GA_SLHH.cpp
)

set(HEADERS
    include/Multimethod.h
    include/Population.h
    include/Problems.h
    include/Workspace.h
    include/CC_HIHH.h
    include/GA_SLHH.h
    include/Rng.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Debug option for nearest-device diagnostics
option(DEBUG_NEAREST "Enable nearest-device debug output" OFF)
if (DEBUG_NEAREST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_NEAREST)
endif()

# Profiling option for evaluation stages
option(PROFILE_EVAL "Enable detailed eval profiling" OFF)
if (PROFILE_EVAL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE_EVAL)
endif()

# MSVC-specific settings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /O2
        /EHsc
        /W3
        /utf-8
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _USE_MATH_DEFINES
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O2
        -Wall
        -Wextra
    )
endif()

# Copy data files to build directory for convenience
file(GLOB DATA_FILES "${CMAKE_SOURCE_DIR}/data/*.txt")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/data")
file(COPY ${DATA_FILES} DESTINATION "${CMAKE_BINARY_DIR}/data")
