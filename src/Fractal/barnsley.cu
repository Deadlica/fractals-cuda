#include <Fractal/barnsley.cuh>
#include <cstdint>

// Simple xorshift32 per-thread PRNG
__device__ inline unsigned int xorshift32(unsigned int& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__global__ void barnsley_clear(unsigned int* accum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) accum[i] = 0;
}

__global__ void barnsley_accum(unsigned int* accum, int width, int height,
                               double x_min, double x_max, double y_min, double y_max,
                               int samples_per_thread, unsigned int seed_base) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int s = seed_base + tid * 2654435761u;
    if (s == 0) s = 1;

    // Skip a handful of iterations to converge onto the attractor.
    double x = 0.0, y = 0.0;
    for (int i = 0; i < 20; i++) {
        unsigned int r = xorshift32(s);
        double p = (r & 0xFFFFFF) / 16777216.0;  // 0..1
        double nx, ny;
        if (p < 0.01) {
            nx = 0.0;                 ny = 0.16 * y;
        } else if (p < 0.86) {
            nx = 0.85 * x + 0.04 * y; ny = -0.04 * x + 0.85 * y + 1.6;
        } else if (p < 0.93) {
            nx = 0.20 * x - 0.26 * y; ny = 0.23 * x + 0.22 * y + 1.6;
        } else {
            nx = -0.15 * x + 0.28 * y; ny = 0.26 * x + 0.24 * y + 0.44;
        }
        x = nx; y = ny;
    }

    for (int i = 0; i < samples_per_thread; i++) {
        unsigned int r = xorshift32(s);
        double p = (r & 0xFFFFFF) / 16777216.0;
        double nx, ny;
        if (p < 0.01) {
            nx = 0.0;                 ny = 0.16 * y;
        } else if (p < 0.86) {
            nx = 0.85 * x + 0.04 * y; ny = -0.04 * x + 0.85 * y + 1.6;
        } else if (p < 0.93) {
            nx = 0.20 * x - 0.26 * y; ny = 0.23 * x + 0.22 * y + 1.6;
        } else {
            nx = -0.15 * x + 0.28 * y; ny = 0.26 * x + 0.24 * y + 0.44;
        }
        x = nx; y = ny;

        if (x >= x_min && x < x_max) {
            int px = static_cast<int>((x - x_min) / (x_max - x_min) * width);
            double py_f = (y - y_min) / (y_max - y_min) * height;
            int py = static_cast<int>(py_f);
            if (px >= 0 && px < width && py >= 0 && py < height) {
                atomicAdd(&accum[py * width + px], 1u);
            }
        }
    }
}

__global__ void barnsley_colourize(uchar4* d_image, const unsigned int* accum,
                                   const Color* palette, int palette_size,
                                   int width, int height, float log_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = width * height;
    if (i >= n) return;
    unsigned int c = accum[i];
    if (c == 0) {
        d_image[i] = make_uchar4(0, 0, 0, 255);
        return;
    }
    float t = log1pf((float)c) / log_max;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    int idx = static_cast<int>(t * (palette_size - 1));
    Color p = palette[idx];
    d_image[i] = make_uchar4(p.r, p.g, p.b, 255);
}

// Accumulator buffer reused across generate calls
static unsigned int* g_accum = nullptr;
static int g_accum_capacity = 0;

static unsigned int* accum_buffer(int n) {
    if (n > g_accum_capacity) {
        if (g_accum) CUDA_CHECK(cudaFree(g_accum));
        CUDA_CHECK(cudaMalloc(&g_accum, n * sizeof(unsigned int)));
        g_accum_capacity = n;
    }
    return g_accum;
}

barnsley::barnsley() : fractal() {}

void barnsley::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);
    unsigned int* accum = accum_buffer(n);

    // Clear
    {
        int bs = 256, gs = (n + bs - 1) / bs;
        barnsley_clear<<<gs, bs>>>(accum, n);
    }

    // Accumulate. Sample count scales with the visible fraction of the
    // attractor so that zoomed-in views don't go nearly black.
    int threads_total  = 1 << 14;                                   // 16k threads
    constexpr double default_area = 6.0 * 11.0;                     // matches default viewport
    double cur_area = fabs((params.x_max - params.x_min) * (params.y_max - params.y_min));
    double scale = default_area / fmax(cur_area, 1e-12);
    if (scale < 1.0) scale = 1.0;
    if (scale > 64.0) scale = 64.0;
    int samples_per = static_cast<int>(512.0 * scale);
    {
        int bs = 256, gs = (threads_total + bs - 1) / bs;
        unsigned int seed = static_cast<unsigned int>(params.max_iter ^ static_cast<int>(params.x_min * 1e6));
        barnsley_accum<<<gs, bs>>>(accum, params.width, params.height,
                                   params.x_min, params.x_max, params.y_min, params.y_max,
                                   samples_per, seed ? seed : 1u);
    }

    // Log-scale: expected density = (points that hit the viewport) / pixels.
    // Assume roughly `1 / scale` of points land in-view; but because we already
    // scaled samples by `scale`, on-attractor density stays ~constant.
    double total_points = static_cast<double>(threads_total) * samples_per;
    double on_attractor_fraction = 1.0 / scale;  // rough
    double expected = total_points * on_attractor_fraction / static_cast<double>(n) * 10.0;
    if (expected < 10.0) expected = 10.0;
    float log_max = logf(1.0f + (float)expected);

    // Colourize
    {
        int bs = 256, gs = (n + bs - 1) / bs;
        barnsley_colourize<<<gs, bs>>>(d_image, accum, PALETTE, PALETTE_SIZE,
                                       params.width, params.height, log_max);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
