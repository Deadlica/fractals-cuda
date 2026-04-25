#ifndef NEWTON_CUH
#define NEWTON_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void newton_kernel(uchar4* d_image, const Color* palette, int palette_size,
                              int width, int height,
                              double x_min, double x_max, double y_min, double y_max,
                              int max_iter, bool smooth, double tolerance);

class newton : public fractal {
public:
    newton();

    void generate(const FractalParams& params) override;
};

#endif // NEWTON_CUH
