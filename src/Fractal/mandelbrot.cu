// Project
#include <Fractal/mandelbrot.cuh>

__global__ void mandelbrot_kernel(uchar4* d_image, const Color* palette, int palette_size,
                                  int width, int height,
                                  double x_min, double x_max, double y_min, double y_max,
                                  int max_iter, bool smooth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    double x0 = x_min + idx * (x_max - x_min) / width;
    double y0 = y_min + idy * (y_max - y_min) / height;
    double x = 0.0, y = 0.0, x2 = 0.0, y2 = 0.0;
    int iter = 0;
    while (x2 + y2 <= 4.0 && iter < max_iter) {
        y = 2.0 * x * y + y0;
        x = x2 - y2 + x0;
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

mandelbrot::mandelbrot(): fractal() {}

void mandelbrot::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    dim3 block(16, 16);
    dim3 grid((params.width + block.x - 1) / block.x,
              (params.height + block.y - 1) / block.y);

    mandelbrot_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, params.smooth
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
