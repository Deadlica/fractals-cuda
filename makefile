compile: src/mandelbrot.cu src/palette.cu src/cli.cpp
	nvcc -O3 -o mandelbrot src/mandelbrot.cu src/palette.cu src/cli.cpp -lopencv_videoio -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

run: compile
	./mandelbrot

clean:
	rm mandelbrot
