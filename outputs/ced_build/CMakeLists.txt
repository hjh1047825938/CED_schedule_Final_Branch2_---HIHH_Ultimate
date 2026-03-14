cmake_minimum_required(VERSION 3.15)
project(CED_Schedule LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    main.cpp
    Multimethod.cpp
    Problems.cpp
    CC_HIHH.cpp
)

set(HEADERS
    Multimethod.h
    Population.h
    Problems.h
    Workspace.h
    CC_HIHH.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Debug option for nearest-device diagnostics
option(DEBUG_NEAREST "Enable nearest-device debug output" OFF)
if (DEBUG_NEAREST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_NEAREST)
endif()

# MSVC-specific settings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /O2         # Optimize for speed
        /EHsc       # Exception handling
        /W3         # Warning level 3
        /utf-8      # Source and execution charset
    )
    # Define _USE_MATH_DEFINES for M_PI and NOMINMAX to avoid min/max macro conflicts
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _USE_MATH_DEFINES
        NOMINMAX
        _CRT_SECURE_NO_WARNINGS
    )
else()
    # GCC/Clang settings
    target_compile_options(${PROJECT_NAME} PRIVATE
        -O2
        -Wall
        -Wextra
    )
endif()

# Copy data files to build directory for convenience
file(GLOB DATA_FILES "*.txt" "data/*.txt")
foreach(DATA_FILE ${DATA_FILES})
    get_filename_component(DATA_FILE_NAME ${DATA_FILE} NAME)
    configure_file(${DATA_FILE} ${CMAKE_BINARY_DIR}/${DATA_FILE_NAME} COPYONLY)
endforeach()
