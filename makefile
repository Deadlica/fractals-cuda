CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++14 -O3
NVCCFLAGS := -arch=sm_50 -Wno-deprecated-gpu-targets

# Include directories
INCLUDES := -Iinclude -I/usr/include/SFML -I/usr/local/cuda/include

# SFML libraries
LIBS := -lsfml-graphics -lsfml-window -lsfml-system

# CUDA libraries
CUDA_LIBS := -L/usr/local/cuda/lib64 -lcuda -lcudart

# Source files
SRC_CPP := src/main.cpp \
	   src/CLI/cli.cpp \
	   src/GUI/add_pattern.cpp \
	   src/GUI/animation.cpp \
           src/GUI/app.cpp \
	   src/GUI/coordinate_label.cpp \
	   src/GUI/menu.cpp \
           src/Util/globals.cpp \
	   src/Util/util.cpp

SRC_CU := src/Fractal/fractal.cu \
	  src/Fractal/burning_ship.cu \
	  src/Fractal/julia.cu \
          src/Fractal/mandelbrot.cu \
	  src/Fractal/newton.cu \
	  src/Fractal/palette.cu \
          src/Fractal/sierpinski.cu

# Object files
OBJ_CPP := $(SRC_CPP:.cpp=.o)
OBJ_CU := $(SRC_CU:.cu=.o)

# Target executable
TARGET := fractals

# Build rules
all: $(TARGET)

$(TARGET): $(OBJ_CPP) $(OBJ_CU)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS) $(CUDA_LIBS)
	rm -f $(OBJ_CPP) $(OBJ_CU)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm $(TARGET)
