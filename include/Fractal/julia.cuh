#ifndef JULIA_CUH
#define JULIA_CUH

// Project
#include <Fractal/fractal.cuh>

__global__ void julia_kernel(Color* d_image, Color* PALETTE,
                                    int* palette_size, int width, int height,
                                    double x_min, double x_max, double y_min,
                                    double y_max, int max_iter,
                                    bool smooth = true);

class julia : public fractal {
public:
    julia();

    void generate(const FractalParams& params) override;
};

#endif // JULIA_CUH
