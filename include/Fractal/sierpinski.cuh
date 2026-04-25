#ifndef SIERPINSKI_CUH
#define SIERPINSKI_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void sierpinski_kernel(uchar4* image, int width, int height,
                                  int depth,
                                  double x_min, double x_max,
                                  double y_min, double y_max);

class sierpinski : public fractal {
public:
    sierpinski();

    void generate(const FractalParams& params) override;
};

#endif // SIERPINSKI_CUH
