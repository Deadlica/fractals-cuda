#ifndef BURNING_SHIP_CUH
#define BURNING_SHIP_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void burning_ship_kernel(uchar4* d_image, const Color* palette, int palette_size,
                                    int width, int height,
                                    double x_min, double x_max, double y_min, double y_max,
                                    int max_iter, bool smooth, int step);

class burning_ship : public fractal {
public:
    burning_ship();

    void generate(const FractalParams& params) override;
};

#endif // BURNING_SHIP_CUH
