// Project
#include <Fractal/julia.cuh>

__global__ void julia_kernel(uchar4* d_image, const Color* palette, int palette_size,
                             int width, int height,
                             double x_min, double x_max, double y_min, double y_max,
                             int max_iter, double c_re, double c_im, bool smooth, int step) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int px = bx * step;
    int py = by * step;
    if (px >= width || py >= height) return;

    double x = x_min + px * (x_max - x_min) / width;
    double y = y_min + py * (y_max - y_min) / height;
    double x2 = x * x, y2 = y * y;
    int iter = 0;
    while (x2 + y2 <= 4.0 && iter < max_iter) {
        y = 2.0 * x * y + c_im;
        x = x2 - y2 + c_re;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }

    uchar4 color;
    if (iter >= max_iter) {
        color = make_uchar4(0, 0, 0, 255);
    } else if (smooth) {
        color = smooth_color(iter, x2 + y2, palette, palette_size);
    } else {
        Color c = palette[iter % palette_size];
        color = make_uchar4(c.r, c.g, c.b, 255);
    }

    int x_end = min(px + step, width);
    int y_end = min(py + step, height);
    for (int yy = py; yy < y_end; yy++)
        for (int xx = px; xx < x_end; xx++)
            d_image[yy * width + xx] = color;
}

julia::julia() : fractal() {}

void julia::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    int step = params.step > 0 ? params.step : 1;
    int grid_w = (params.width  + step - 1) / step;
    int grid_h = (params.height + step - 1) / step;

    dim3 block(16, 16);
    dim3 grid((grid_w + block.x - 1) / block.x,
              (grid_h + block.y - 1) / block.y);

    double c_re = params.c_re;
    double c_im = params.c_im;

    julia_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, c_re, c_im, params.smooth, step
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
