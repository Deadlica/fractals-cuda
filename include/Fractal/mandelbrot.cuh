#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void mandelbrot_kernel(Color* d_image, Color* PALETTE,
                                  int* palette_size, int width, int height,
                                  double x_min, double x_max, double y_min,
                                  double y_max, int max_iter,
                                  bool smooth = true);

class mandelbrot : public fractal {
public:
    mandelbrot();

    void generate(const FractalParams& params) override;
};

#endif // MANDELBROT_CUH
