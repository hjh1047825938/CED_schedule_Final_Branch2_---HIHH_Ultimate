cmake_minimum_required(VERSION 3.15)
project(CED_Schedule LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    src/main.cpp
    src/Multimethod.cpp
    src/Problems.cpp
    src/CC_HIHH_singleisland.cpp
    src/GA_SLHH.cpp
    src/QPHH.cpp
    src/IMOMA.cpp
    src/CGA.cpp
)

set(CORE_SOURCES
    src/Multimethod.cpp
    src/Problems.cpp
)

set(HEADERS
    include/Multimethod.h
    include/Population.h
    include/Problems.h
    include/Workspace.h
    include/CC_HIHH.h
    include/GA_SLHH.h
    include/QPHH.h
    include/IMOMA.h
    include/CGA.h
    include/Rng.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
add_executable(ppo_scheduler
    src/ppo_scheduler_main.cpp
    src/PPO.cpp
    ${CORE_SOURCES}
    include/PPO.h
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(ppo_scheduler PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(ppo_scheduler PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled for ${PROJECT_NAME}")
else()
    message(STATUS "OpenMP not found; building without parallel pragmas")
endif()

# Debug option for nearest-device diagnostics
option(DEBUG_NEAREST "Enable nearest-device debug output" OFF)
if (DEBUG_NEAREST)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_NEAREST)
    target_compile_definitions(ppo_scheduler PRIVATE DEBUG_NEAREST)
endif()

# Profiling option for evaluation stages
option(PROFILE_EVAL "Enable detailed eval profiling" OFF)
if (PROFILE_EVAL)
    target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE_EVAL)
    target_compile_definitions(ppo_scheduler PRIVATE PROFILE_EVAL)
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
    target_compile_options(ppo_scheduler PRIVATE
        /O2
        /EHsc
        /W3
        /utf-8
    )
    target_compile_definitions(ppo_scheduler PRIVATE
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
    target_compile_options(ppo_scheduler PRIVATE
        -O2
        -Wall
        -Wextra
    )
endif()

# Copy data files to build directory for convenience
file(GLOB DATA_FILES "${CMAKE_SOURCE_DIR}/data/*.txt")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/data")
file(COPY ${DATA_FILES} DESTINATION "${CMAKE_BINARY_DIR}/data")
