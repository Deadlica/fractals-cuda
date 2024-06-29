cmake_minimum_required(VERSION 3.20)
project(mandelbrot)

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES src/main.cpp src/cli.cpp src/add_pattern.cpp src/mandelbrot.cu src/palette.cu)

# Set SFML root path for WSL
set(SFML_ROOT /usr/include/SFML)

# Find SFML package
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set CUDA compilation flags to a valid architecture (e.g., sm_50 for Maxwell)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50")

# Add CUDA source files
cuda_add_library(mandelbrot_cuda src/mandelbrot.cu src/palette.cu)

# Add the executable
add_executable(mandelbrot ${SOURCE_FILES})

# Link SFML libraries
target_link_libraries(mandelbrot sfml-system sfml-window sfml-graphics)

# Link CUDA libraries
target_link_libraries(mandelbrot ${CUDA_LIBRARIES} mandelbrot_cuda)

# Include directories for SFML and CUDA
target_include_directories(mandelbrot PRIVATE ${SFML_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# Set properties for the CUDA library
set_target_properties(mandelbrot_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
