compile: src/main.cpp src/mandelbrot.cu src/palette.cu src/cli.cpp src/add_pattern.cpp
	nvcc -O3 -o mandelbrot src/main.cpp src/cli.cpp src/mandelbrot.cu src/palette.cu src/add_pattern.cpp -lsfml-graphics -lsfml-window -lsfml-system

run: compile
	./mandelbrot

clean:
	rm mandelbrot
