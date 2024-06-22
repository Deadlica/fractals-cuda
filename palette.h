#ifndef PALETTE_H
#define PALETTE_H

#include <cuda_runtime.h>

struct Color {
    unsigned char r, g, b;
    __host__ __device__ Color() : r(0), g(0), b(0) {}
    __host__ __device__ Color(unsigned char _r, unsigned char _g, unsigned char _b) : r(_r), g(_g), b(_b) {}

    __host__ __device__ unsigned int toInt() const {
        return (r << 16) | (g << 8) | b;
    }
};

extern Color* PALETTE;
extern int* PALETTE_SIZE;

void initialize_palette();
void free_palette();

#endif // PALETTE_H

