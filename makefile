compile: src/main.cpp src/mandelbrot.cu src/palette.cu src/cli.cpp
	nvcc -O3 -o mandelbrot src/main.cpp src/cli.cpp src/mandelbrot.cu src/palette.cu -lsfml-graphics -lsfml-window -lsfml-system

run: compile
	./mandelbrot

clean:
	rm mandelbrot
