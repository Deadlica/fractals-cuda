#include <Fractal/lyapunov.cuh>

__global__ void lyapunov_kernel(uchar4* d_image, const Color* palette, int palette_size,
                                int width, int height,
                                double a_min, double a_max, double b_min, double b_max,
                                int max_iter) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    double a = a_min + px * (a_max - a_min) / width;
    double b = b_min + py * (b_max - b_min) / height;

    double x = 0.5;
    int warm = max_iter / 4;
    for (int i = 0; i < warm; i++) {
        double r = (i & 1) ? b : a;
        x = r * x * (1.0 - x);
    }
    double sum = 0.0;
    int count = max_iter - warm;
    for (int i = 0; i < count; i++) {
        double r = (i & 1) ? b : a;
        x = r * x * (1.0 - x);
        double d = r * (1.0 - 2.0 * x);
        sum += log(fabs(d) + 1e-300);
    }
    double lambda = sum / count;

    uchar4 color;
    if (!isfinite(lambda)) {
        color = make_uchar4(0, 0, 0, 255);
    } else {
        // Map lambda from [-2, 1] into palette index. Negative (stable) ->
        // low indices, positive (chaotic) -> high indices.
        double t = (lambda + 2.0) / 3.0;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double idxf = t * (palette_size - 1);
        int i0 = static_cast<int>(floor(idxf));
        int i1 = i0 + 1 < palette_size ? i0 + 1 : i0;
        double f = idxf - i0;
        Color c1 = palette[i0];
        Color c2 = palette[i1];
        color = lerp_rgba(c1, c2, f);
    }
    d_image[py * width + px] = color;
}

lyapunov::lyapunov() : fractal() {}

void lyapunov::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    dim3 block(16, 16);
    dim3 grid((params.width + block.x - 1) / block.x,
              (params.height + block.y - 1) / block.y);

    lyapunov_kernel<<<grid, block>>>(d_image, PALETTE, PALETTE_SIZE,
                                     params.width, params.height,
                                     params.x_min, params.x_max, params.y_min, params.y_max,
                                     params.max_iter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
