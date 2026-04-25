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
    src/GUI/options_page.cpp
    src/GUI/Widgets/checkbox.cpp
    src/GUI/Widgets/dropdown.cpp
    src/GUI/Widgets/numeric_input_field.cpp
    src/Util/globals.cpp
    src/Util/util.cpp
    include/Fractal/Params/FractalParams.h
)

set(SFML_ROOT /usr/include/SFML)
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86 -Wno-deprecated-gpu-targets")

cuda_add_library(fractals_cuda
                 src/Fractal/fractal.cu
                 src/Fractal/burning_ship.cu
                 src/Fractal/julia.cu
                 src/Fractal/mandelbrot.cu
                 src/Fractal/newton.cu
                 src/Fractal/palette.cu
                 src/Fractal/sierpinski.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fractals ${SOURCE_FILES})

target_link_libraries(fractals sfml-system sfml-window sfml-graphics)
target_link_libraries(fractals ${CUDA_LIBRARIES} fractals_cuda)

target_include_directories(fractals PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${SFML_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(fractals_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/include"
)
