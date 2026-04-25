#ifndef FRACTAL_TYPE_H
#define FRACTAL_TYPE_H

// std
#include <string>

enum class fractal_type {
    MANDELBROT, NEWTON, BURNING_SHIP, JULIA, SIERPINSKI
};

struct viewport { double x_min, x_max, y_min, y_max; };

inline viewport default_viewport(fractal_type t) {
    switch (t) {
        case fractal_type::MANDELBROT:   return {-2.0,  1.0, -1.5, 1.5};
        case fractal_type::BURNING_SHIP: return {-2.5,  1.5, -2.0, 1.0};
        case fractal_type::JULIA:        return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::NEWTON:       return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::SIERPINSKI:   return {-0.8,  0.8, -0.05, 0.75};
    }
    return {-2.0, 1.0, -1.5, 1.5};
}

inline std::string fractal_type_to_string(fractal_type t) {
    switch (t) {
        case fractal_type::MANDELBROT:   return "mandelbrot";
        case fractal_type::BURNING_SHIP: return "burning_ship";
        case fractal_type::JULIA:        return "julia";
        case fractal_type::NEWTON:       return "newton";
        case fractal_type::SIERPINSKI:   return "sierpinski";
    }
    return "mandelbrot";
}

inline bool fractal_type_from_string(const std::string& s, fractal_type& out) {
    if (s == "mandelbrot")        { out = fractal_type::MANDELBROT;   return true; }
    if (s == "burning_ship")      { out = fractal_type::BURNING_SHIP; return true; }
    if (s == "julia")             { out = fractal_type::JULIA;        return true; }
    if (s == "newton")            { out = fractal_type::NEWTON;       return true; }
    if (s == "sierpinski")        { out = fractal_type::SIERPINSKI;   return true; }
    return false;
}

#endif // FRACTAL_TYPE_H
