#ifndef NOVA_CUH
#define NOVA_CUH

#include <Fractal/fractal.cuh>

__global__ void nova_kernel(uchar4* d_image, const Color* palette, int palette_size,
                            int width, int height,
                            double x_min, double x_max, double y_min, double y_max,
                            int max_iter, double tolerance, bool smooth, int step);

class nova : public fractal {
public:
    nova();
    void generate(const FractalParams& params) override;
};

#endif
