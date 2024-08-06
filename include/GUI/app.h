#ifndef APP_H
#define APP_H

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
    Color* h_image;
};

class app {

};


#endif // APP_H
