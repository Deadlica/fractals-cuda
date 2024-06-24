#ifndef MANDELBROT_H
#define MANDELBROT_H

#include "palette.h"

__device__ Color linear_interpolate(const Color& color1, const Color& color2, double t);

__global__ void mandelbrot_kernel(Color* d_image, Color* PALETTE,
                                 int* palette_size, int width, int height,
                                 double x_min, double x_max, double y_min,
                                 double y_max, int max_iter,
                                 bool smooth = true);

void mandelbrot(Color* h_image, int width, int height, double x_min,
                double x_max, double y_min, double y_max, int max_iter, bool smooth);



#endif
