#include <Fractal/sierpinski.cuh>

// Recursive subdivision test for membership in the Sierpinski gasket filling
// the unit right-triangle {(u,v) : u >= 0, v >= 0, u + v <= 1}.
//
// At each level the triangle is split into four half-size sub-triangles;
// the middle (upside-down) one is empty. Zooming in just requires more
// levels, so depth scales with zoom without grid-resolution limits.
__device__ bool in_sierpinski(double u, double v, int depth) {
    if (u < 0.0 || v < 0.0 || u + v > 1.0) return false;
    for (int i = 0; i < depth; i++) {
        u *= 2.0;
        v *= 2.0;
        if (u >= 1.0 && v <= 1.0) {            // right sub-triangle
            u -= 1.0;
        } else if (v >= 1.0 && u <= 1.0) {     // top sub-triangle
            v -= 1.0;
        } else if (u + v <= 1.0) {             // bottom-left sub-triangle
            // no offset
        } else {                               // center (upside-down) hole
            return false;
        }
    }
    return true;
}

__global__ void sierpinski_kernel(uchar4* image, int width, int height,
                                  int depth,
                                  double x_min, double x_max,
                                  double y_min, double y_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Pixel -> viewport coordinates.
    double px = x_min + x * (x_max - x_min) / width;
    double py = y_min + y * (y_max - y_min) / height;

    // Rotate -45 degrees (i.e. 45 deg CW in screen space where y grows down).
    // This puts the triangle's apex at the top and its hypotenuse along the
    // bottom, which reads as "upright" to a human.
    constexpr double INV_SQRT2 = 0.70710678118654752440;
    double u =  (px + py) * INV_SQRT2;
    double v = (-px + py) * INV_SQRT2;

    uchar4 color = make_uchar4(0, 0, 0, 255);
    int d = depth < 1 ? 1 : (depth > 50 ? 50 : depth);
    if (in_sierpinski(u, v, d)) color = make_uchar4(255, 255, 255, 255);
    image[y * width + x] = color;
}

sierpinski::sierpinski(): fractal() {}

void sierpinski::generate(const FractalParams& params) {
    int n = params.width * params.height;
    uchar4* d_image = device_image_buffer(n);

    dim3 block(16, 16);
    dim3 grid((params.width + block.x - 1) / block.x,
              (params.height + block.y - 1) / block.y);

    sierpinski_kernel<<<grid, block>>>(
        d_image, params.width, params.height, params.depth,
        params.x_min, params.x_max, params.y_min, params.y_max
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(params.h_image, d_image, n * sizeof(uchar4), cudaMemcpyDeviceToHost));
}
