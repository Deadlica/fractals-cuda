#include <Fractal/fractal.cuh>

fractal::fractal() = default;

static uchar4* g_d_image = nullptr;
static int g_capacity = 0;

uchar4* device_image_buffer(int n) {
    if (n > g_capacity) {
        if (g_d_image) CUDA_CHECK(cudaFree(g_d_image));
        CUDA_CHECK(cudaMalloc(&g_d_image, n * sizeof(uchar4)));
        g_capacity = n;
    }
    return g_d_image;
}

void free_device_image_buffer() {
    if (g_d_image) {
        cudaFree(g_d_image);
        g_d_image = nullptr;
        g_capacity = 0;
    }
}
