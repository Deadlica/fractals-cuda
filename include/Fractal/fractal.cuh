#ifndef FRACTAL_CUH
#define FRACTAL_CUH

// Project
#include <Fractal/Params/FractalParams.h>
#include <Fractal/palette.cuh>

// CUDA
#include <cuda_runtime.h>

__device__ inline Color linear_interpolate(const Color& color1, const Color& color2, double t) {
    unsigned char r = static_cast<unsigned char>(color1.r + t * (color2.r - color1.r));
    unsigned char g = static_cast<unsigned char>(color1.g + t * (color2.g - color1.g));
    unsigned char b = static_cast<unsigned char>(color1.b + t * (color2.b - color1.b));
    return Color(r, g, b);
}

class fractal {
public:
    fractal();
    virtual ~fractal() = default;

    virtual void generate(const FractalParams& params) = 0;
};

#endif // FRACTAL_CUH
