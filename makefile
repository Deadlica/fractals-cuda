compile: mandelbrot.cu palette.cu
	nvcc -o mandelbrot mandelbrot.cu palette.cu -lopencv_videoio -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

run: compile
	./mandelbrot

clean:
	rm mandelbrot
