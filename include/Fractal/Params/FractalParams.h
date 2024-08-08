#ifndef FRACTALPARAMS_H
#define FRACTALPARAMS_H

// Project
#include <Fractal/palette.cuh>

struct FractalParams {
    int width;
    int height;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    int max_iter;
    double zoom_factor;
    bool smooth;
    double tolerance;
    int depth;
    Color* h_image;
};

#endif // FRACTALPARAMS_H
