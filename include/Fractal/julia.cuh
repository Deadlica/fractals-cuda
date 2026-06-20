#ifndef JULIA_CUH
#define JULIA_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void julia_kernel(uchar4* d_image, const Color* palette, int palette_size,
                             int width, int height,
                             double x_min, double x_max, double y_min, double y_max,
                             int max_iter, double c_re, double c_im, bool smooth, int step);

class julia : public fractal {
public:
    julia();

    void generate(const FractalParams& params) override;
};

#endif // JULIA_CUH
