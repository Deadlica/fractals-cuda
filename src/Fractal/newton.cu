// Project
#include <Fractal/newton.cuh>

__device__ double2 complex_add(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ double2 complex_sub(double2 a, double2 b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ double2 complex_mul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ double2 complex_div(double2 a, double2 b) {
    double denominator = b.x * b.x + b.y * b.y;
    return make_double2((a.x * b.x + a.y * b.y) / denominator, (a.y * b.x - a.x * b.y) / denominator);
}

__device__ double2 complex_pow(double2 a, int n) {
    double2 result = make_double2(1.0, 0.0);
    for (int i = 0; i < n; i++) {
        result = complex_mul(result, a);
    }
    return result;
}

__global__ void newton_kernel(Color* d_image, Color* PALETTE,
                              int* palette_size, int width, int height,
                              double x_min, double x_max, double y_min,
                              double y_max, int max_iter,
                              bool smooth, double tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) {
        return;
    }

    double x0 = x_min + idx * (x_max - x_min) / width;
    double y0 = y_min + idy * (y_max - y_min) / height;
    double2 z = make_double2(x0, y0);
    int iter = 0;

    while (iter < max_iter) {
        double2 z_n = complex_pow(z, 3);  // z^3
        double2 f_z = complex_sub(z_n, make_double2(1.0, 0.0));  // f(z) = z^3 - 1
        double2 df_z = complex_mul(make_double2(3.0, 0.0), complex_pow(z, 2));  // f'(z) = 3z^2
        double2 z_next = complex_sub(z, complex_div(f_z, df_z));

        if (sqrtf((z_next.x - z.x) * (z_next.x - z.x) + (z_next.y - z.y) * (z_next.y - z.y)) < tolerance) {
            break;
        }

        z = z_next;
        iter++;
    }

    if (smooth) {
        double t = static_cast<double>(iter) / max_iter;
        Color color1 = PALETTE[iter % *palette_size];
        Color color2 = PALETTE[(iter + 1) % *palette_size];
        Color color = linear_interpolate(color1, color2, t);
        d_image[idy * width + idx] = color;
    } else {
        d_image[idy * width + idx] = PALETTE[iter % *palette_size];
    }
}

newton::newton() : fractal() {}

void newton::generate(const FractalParams& params) {
    Color* d_image;
    size_t image_size = params.width * params.height * sizeof(Color);
    cudaMalloc(&d_image, image_size);

    dim3 block_size(32, 32);
    dim3 grid_size(
            (params.width + block_size.x - 1) / block_size.x,
            (params.height + block_size.y - 1) / block_size.y
    );

    newton_kernel<<<grid_size, block_size>>>(
            d_image, PALETTE, PALETTE_SIZE, params.width, params.height, params.x_min, params.x_max,
            params.y_min, params.y_max, params.max_iter, params.smooth, params.tolerance
    );

    cudaMemcpy(params.h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
}
