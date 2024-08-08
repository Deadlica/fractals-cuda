#ifndef SIERPINSKI_CUH
#define SIERPINSKI_CUH

// Project
#include <Fractal/fractal.cuh>

__device__ int inside_sierpinski_triangle(int x, int y, int depth);

__global__ void sierpinski_kernel(Color* image, int width, int height, int depth);

class sierpinski : public fractal {
public:
    sierpinski();

    void generate(const FractalParams& params) override;
};

#endif // SIERPINSKI_CUH
