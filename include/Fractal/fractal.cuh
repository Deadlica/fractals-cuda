#ifndef FRACTAL_CUH
#define FRACTAL_CUH

// Project
#include <Fractal/Params/FractalParams.h>
#include <Fractal/palette.cuh>

// CUDA
#include <cuda_runtime.h>

// std
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do {                                               \
    cudaError_t _err = (call);                                              \
    if (_err != cudaSuccess) {                                              \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                     __FILE__, __LINE__, cudaGetErrorString(_err));         \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)

__device__ inline uchar4 lerp_rgba(const Color& a, const Color& b, double t) {
    return make_uchar4(
        static_cast<unsigned char>(a.r + t * (b.r - a.r)),
        static_cast<unsigned char>(a.g + t * (b.g - a.g)),
        static_cast<unsigned char>(a.b + t * (b.b - a.b)),
        255
    );
}

// Shared Mandelbrot-family smooth colouring: given escape-time iter (< max),
// current |z|^2, and palette, produce an RGBA pixel.
__device__ inline uchar4 smooth_color(int iter, double mod_sq,
                                      const Color* palette, int palette_size) {
    // Normalized iteration count: iter + 1 - log2(log2(|z|))
    double log_zn = log(mod_sq) * 0.5;
    double nu = log(log_zn / log(2.0)) / log(2.0);
    double iter_d = iter + 1 - nu;
    int i = static_cast<int>(floor(iter_d));
    double t = iter_d - i;
    Color c1 = palette[((i % palette_size) + palette_size) % palette_size];
    Color c2 = palette[(((i + 1) % palette_size) + palette_size) % palette_size];
    return lerp_rgba(c1, c2, t);
}

// Grow-only device image buffer, reused across frames.
uchar4* device_image_buffer(int n);
void free_device_image_buffer();

class fractal {
public:
    fractal();
    virtual ~fractal() = default;

    virtual void generate(const FractalParams& params) = 0;
};

#endif // FRACTAL_CUH
