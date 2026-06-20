#ifndef FRACTALPARAMS_H
#define FRACTALPARAMS_H

// CUDA
#include <vector_types.h>

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
    double c_re;
    double c_im;
    int multibrot_n;
    int step;   // pixel block size (1 = full quality)
    uchar4* h_image;
};

#endif // FRACTALPARAMS_H
