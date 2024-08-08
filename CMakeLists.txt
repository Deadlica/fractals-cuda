cmake_minimum_required(VERSION 3.20)
project(fractals)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SOURCE_FILES
    src/main.cpp
    src/CLI/cli.cpp
    src/GUI/add_pattern.cpp
    src/GUI/animation.cpp
    src/GUI/app.cpp
    src/GUI/coordinate_label.cpp
    src/GUI/menu.cpp
    src/Util/globals.cpp
    src/Util/util.cpp
    include/Fractal/Params/FractalParams.h
)

# Set SFML root path for WSL
set(SFML_ROOT /usr/include/SFML)

# Find SFML package
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA compilation flags to a valid architecture (e.g., sm_50 for Maxwell)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -Wno-deprecated-gpu-targets")

# Add CUDA source files
cuda_add_library(fractals_cuda
                 src/Fractal/fractal.cu
                 src/Fractal/burning_ship.cu
                 src/Fractal/julia.cu
                 src/Fractal/mandelbrot.cu
                 src/Fractal/newton.cu
                 src/Fractal/palette.cu
                 src/Fractal/sierpinski.cu
)

# Add the executable
add_executable(fractals ${SOURCE_FILES})

# Link SFML libraries
target_link_libraries(fractals sfml-system sfml-window sfml-graphics)

# Link CUDA libraries
target_link_libraries(fractals ${CUDA_LIBRARIES} fractals_cuda)

# Include directories for SFML and CUDA
target_include_directories(fractals PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${SFML_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Set properties for the CUDA library
set_target_properties(fractals_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include"
)
