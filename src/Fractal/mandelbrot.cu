// Project
#include <Fractal/mandelbrot.cuh>

__global__ void mandelbrot_kernel(Color* d_image, Color* PALETTE,
                                  int* palette_size, int width, int height,
                                  double x_min, double x_max, double y_min,
                                  double y_max, int max_iter,
                                  bool smooth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) {
        return;
    }

    double x0 = x_min + idx * (x_max - x_min) / width;
    double y0 = y_min + idy * (y_max - y_min) / height;
    double x = 0.0, y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < max_iter) {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iter++;
    }

    if (smooth) {
        double t;
        double iter_d;
        if (iter < max_iter) {
            double log_zn = logf(x * x + y * y) / 2.0f;
            double nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
            iter_d = iter + 1 - nu;
            iter = static_cast<int>(floor(iter_d));
        }
        t = iter_d - iter;
        Color color1 = PALETTE[iter % *palette_size];
        Color color2 = PALETTE[(iter + 1) % *palette_size];
        Color color = linear_interpolate(color1, color2, t);
        d_image[idy * width + idx] = color;
    }
    else {
        d_image[idy * width + idx] = PALETTE[iter % *palette_size];
    }
}

mandelbrot::mandelbrot(): fractal() {}

void mandelbrot::generate(const FractalParams& params) {
    Color* d_image;
    size_t image_size = params.width * params.height * sizeof(Color);
    cudaMalloc(&d_image, image_size);

    dim3 block_size(32, 32);
    dim3 grid_size(
        (params.width + block_size.x - 1) / block_size.x,
        (params.height + block_size.y - 1) / block_size.y
    );

    mandelbrot_kernel<<<grid_size, block_size>>>(
            d_image, PALETTE, PALETTE_SIZE,params.width, params.height,
            params.x_min, params.x_max, params.y_min, params.y_max, params.max_iter, params.smooth
    );

    cudaMemcpy(params.h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
}