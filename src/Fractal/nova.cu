#include <Fractal/nova.cuh>

__device__ inline double2 cmul_n(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
__device__ inline double2 cdiv_n(double2 a, double2 b) {
    double d = b.x * b.x + b.y * b.y;
    return make_double2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

__global__ void nova_kernel(uchar4* d_image, const Color* palette, int palette_size,
                            int width, int height,
                            double x_min, double x_max, double y_min, double y_max,
                            int max_iter, double tolerance, bool smooth, int step) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int px = bx * step;
    int py = by * step;
    if (px >= width || py >= height) return;

    double cr = x_min + px * (x_max - x_min) / width;
    double ci = y_min + py * (y_max - y_min) / height;
    double2 c = make_double2(cr, ci);
    double2 z = make_double2(1.0, 0.0);

    int iter = 0;
    double last_d2 = 1e300;
    double prev_d2 = 1e300;
    bool converged = false;
    while (iter < max_iter) {
        double2 z2 = cmul_n(z, z);
        double2 z3 = cmul_n(z2, z);
        double2 f  = make_double2(z3.x - 1.0, z3.y);
        double2 df = make_double2(3.0 * z2.x, 3.0 * z2.y);
        double2 dz = cdiv_n(f, df);
        double2 zn = make_double2(z.x - dz.x + c.x, z.y - dz.y + c.y);

        double dx = zn.x - z.x, dy = zn.y - z.y;
        prev_d2 = last_d2;
        last_d2 = dx * dx + dy * dy;
        if (last_d2 < tolerance * tolerance) { converged = true; break; }

        z = zn;
        iter++;
    }

    uchar4 color;
    if (smooth && converged && last_d2 > 0.0 && prev_d2 > last_d2 && iter > 0) {
        double log_tol2 = log(tolerance * tolerance);
        double log_last = log(last_d2);
        double log_prev = log(prev_d2);
        double frac = (log_prev - log_tol2) / (log_prev - log_last);
        if (frac < 0.0) frac = 0.0;
        if (frac > 1.0) frac = 1.0;
        double mu = (iter - 1) + frac;
        int base = static_cast<int>(floor(mu));
        double t = mu - base;
        Color c1 = palette[((base % palette_size) + palette_size) % palette_size];
        Color c2 = palette[(((base + 1) % palette_size) + palette_size) % palette_size];
        color = lerp_rgba(c1, c2, t);
    } else {
        Color col = palette[iter % palette_size];
        color = make_uchar4(col.r, col.g, col.b, 255);
    }

    int x_end = min(px + step, width);
    int y_end = min(py + step, height);
    for (int yy = py; yy < y_end; yy++)
        for (int xx = px; xx < x_end; xx++)
            d_image[yy * width + xx] = color;
}

nova::nova() : fractal() {}

void nova::generate(const FractalParams& params) {
    int n_pix = params.width * params.height;
    uchar4* d_image = device_image_buffer(n_pix);

    int step = params.step > 0 ? params.step : 1;
    int grid_w = (params.width  + step - 1) / step;
    int grid_h = (params.height + step - 1) / step;

    dim3 block(16, 16);
    dim3 grid((grid_w + block.x - 1) / block.x,
              (grid_h + block.y - 1) / block.y);

    nova_kernel<<<grid, block>>>(
        d_image, PALETTE, PALETTE_SIZE, params.width, params.height,
        params.x_min, params.x_max, params.y_min, params.y_max,
        params.max_iter, params.tolerance, params.smooth, step
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n_pix * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
