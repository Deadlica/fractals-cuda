#ifndef MULTIBROT_CUH
#define MULTIBROT_CUH

#include <Fractal/fractal.cuh>

__global__ void multibrot_kernel(uchar4* d_image, const Color* palette, int palette_size,
                                 int width, int height,
                                 double x_min, double x_max, double y_min, double y_max,
                                 int max_iter, int n, bool smooth, int step);

class multibrot : public fractal {
public:
    multibrot();
    void generate(const FractalParams& params) override;
};

#endif
