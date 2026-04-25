// Project
#include <Fractal/julia.cuh>

__global__ void julia_kernel(uchar4* d_image, const Color* palette, int palette_size,
                             int width, int height,
                             double x_min, double x_max, double y_min, double y_max,
                             int max_iter, double c_re, double c_im, bool smooth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    double x = x_min + idx * (x_max - x_min) / width;
    double y = y_min + idy * (y_max - y_min) / height;
    double x2 = x * x, y2 = y * y;
    int iter = 0;
    while (x2 + y2 <= 4.0 && iter < max_iter) {
        y = 2.0 * x * y + c_im;
        x = x2 - y2 + c_re;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }

    int pixel = idy * width + idx;
    if (iter >= max_iter) {
        d_image[pixel] = make_uchar4(0, 0, 0, 255);
    } else if (smooth) {
        d_image[pixel] = smooth_color(iter, x2 + y2, palette, palette_size);
    } else {
        Color c = palette[iter % palette_size];
        d_image[pixel] = make_uchar4(c.r, c.g, c.b, 255);
    }
}

julia::julia() : fractal() {}

void julia::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    dim3 block(16, 16);
    dim3 grid((params.width + block.x - 1) / block.x,
              (params.height + block.y - 1) / block.y);

    double c_re = params.c_re;
    double c_im = params.c_im;

    julia_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, c_re, c_im, params.smooth
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
