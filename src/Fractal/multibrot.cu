#include <Fractal/multibrot.cuh>

__global__ void multibrot_kernel(uchar4* d_image, const Color* palette, int palette_size,
                                 int width, int height,
                                 double x_min, double x_max, double y_min, double y_max,
                                 int max_iter, int n, bool smooth, int step) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int px = bx * step;
    int py = by * step;
    if (px >= width || py >= height) return;

    double x0 = x_min + px * (x_max - x_min) / width;
    double y0 = y_min + py * (y_max - y_min) / height;
    double x = 0.0, y = 0.0;
    int iter = 0;
    while (x * x + y * y <= 4.0 && iter < max_iter) {
        double zr = 1.0, zi = 0.0;
        for (int k = 0; k < n; k++) {
            double nr = zr * x - zi * y;
            double ni = zr * y + zi * x;
            zr = nr; zi = ni;
        }
        x = zr + x0;
        y = zi + y0;
        iter++;
    }

    uchar4 color;
    if (iter >= max_iter) {
        color = make_uchar4(0, 0, 0, 255);
    } else if (smooth) {
        color = smooth_color(iter, x * x + y * y, palette, palette_size);
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

multibrot::multibrot() : fractal() {}

void multibrot::generate(const FractalParams& params) {
    int n_pix = params.width * params.height;
    uchar4* d_image = device_image_buffer(n_pix);

    int step = params.step > 0 ? params.step : 1;
    int grid_w = (params.width  + step - 1) / step;
    int grid_h = (params.height + step - 1) / step;

    dim3 block(16, 16);
    dim3 grid((grid_w + block.x - 1) / block.x,
              (grid_h + block.y - 1) / block.y);

    int n = params.multibrot_n;
    if (n < 2) n = 2;
    if (n > 8) n = 8;

    multibrot_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, n, params.smooth, step
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n_pix * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
