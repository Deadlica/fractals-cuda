// Project
#include <Fractal/newton.cuh>

__device__ inline double2 cadd(double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); }
__device__ inline double2 csub(double2 a, double2 b) { return make_double2(a.x - b.x, a.y - b.y); }
__device__ inline double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ inline double2 cdiv(double2 a, double2 b) {
    double d = b.x * b.x + b.y * b.y;
    return make_double2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

__global__ void newton_kernel(uchar4* d_image, const Color* palette, int palette_size,
                              int width, int height,
                              double x_min, double x_max, double y_min, double y_max,
                              int max_iter, bool smooth, double tolerance, int step) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int px = bx * step;
    int py = by * step;
    if (px >= width || py >= height) return;

    double x0 = x_min + px * (x_max - x_min) / width;
    double y0 = y_min + py * (y_max - y_min) / height;
    double2 z = make_double2(x0, y0);
    int iter = 0;

    while (iter < max_iter) {
        double2 z2 = cmul(z, z);
        double2 z3 = cmul(z2, z);
        double2 f  = csub(z3, make_double2(1.0, 0.0));
        double2 df = cmul(make_double2(3.0, 0.0), z2);
        double2 zn = csub(z, cdiv(f, df));

        double dx = zn.x - z.x, dy = zn.y - z.y;
        if (dx * dx + dy * dy < tolerance * tolerance) break;

        z = zn;
        iter++;
    }

    uchar4 color;
    if (smooth) {
        double t = static_cast<double>(iter) / max_iter;
        Color c1 = palette[iter % palette_size];
        Color c2 = palette[(iter + 1) % palette_size];
        color = lerp_rgba(c1, c2, t);
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

newton::newton() : fractal() {}

void newton::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    int step = params.step > 0 ? params.step : 1;
    int grid_w = (params.width  + step - 1) / step;
    int grid_h = (params.height + step - 1) / step;

    dim3 block(16, 16);
    dim3 grid((grid_w + block.x - 1) / block.x,
              (grid_h + block.y - 1) / block.y);

    newton_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, params.smooth, params.tolerance, step
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
