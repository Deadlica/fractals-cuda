#ifndef PALETTE_CUH
#define PALETTE_CUH

// std
#include <string>

// CUDA
#include <cuda_runtime.h>

struct Color {
    unsigned char r, g, b;
    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(unsigned char _r,
                              unsigned char _g,
                              unsigned char _b) : r(_r), g(_g), b(_b) {}

    __host__ __device__ unsigned int to_int() const {
        return (r << 16) | (g << 8) | b;
    }
};

extern Color* PALETTE;
extern int* PALETTE_SIZE;

std::string get_theme_path(const std::string& theme);
void load_color_theme(const std::string& path);
void initialize_palette(const std::string& theme);
void free_palette();

#endif // PALETTE_CUH

