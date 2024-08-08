#include <Fractal/sierpinski.cuh>

__device__ int inside_sierpinski_triangle(int x, int y, int depth) {
    // Check if the point is inside the Sierpinski triangle at the given depth
    while (depth--) {
        if ((x & 1) == 0 && (y & 1) == 0) {
            return 1; // Inside the triangle
        }
        if ((x & 1) == 1 && (y & 1) == 1) {
            return 0; // Outside the triangle
        }
        x >>= 1;
        y >>= 1;
    }
    return 0;
}

__global__ void sierpinski_kernel(Color* image, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Map pixel coordinates to Sierpinski coordinates
        int sx = (x * 3) / width; // Simple mapping for illustration
        int sy = (y * 3) / height;

        // Determine color based on depth
        if (inside_sierpinski_triangle(sx, sy, depth)) {
            image[y * width + x] = Color{255, 255, 255}; // White for inside
        } else {
            image[y * width + x] = Color{0, 0, 0}; // Black for outside
        }
    }
}

sierpinski::sierpinski(): fractal() {}

void sierpinski::generate(const FractalParams& params) {
    Color* d_image;
    size_t image_size = params.width * params.height * sizeof(Color);
    cudaMalloc(&d_image, image_size);

    dim3 block_size(32, 32);
    dim3 grid_size(
            (params.width + block_size.x - 1) / block_size.x,
            (params.height + block_size.y - 1) / block_size.y
    );

    sierpinski_kernel<<<grid_size, block_size>>>(
            d_image, params.width, params.height, params.depth
    );

    cudaMemcpy(params.h_image, d_image, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
}
