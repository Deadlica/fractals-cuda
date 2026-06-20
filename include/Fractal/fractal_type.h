#ifndef FRACTAL_TYPE_H
#define FRACTAL_TYPE_H

// std
#include <string>

enum class fractal_type {
    MANDELBROT, NEWTON, BURNING_SHIP, JULIA, SIERPINSKI,
    MULTIBROT, NOVA, BARNSLEY, LYAPUNOV
};

struct viewport { double x_min, x_max, y_min, y_max; };

inline viewport default_viewport(fractal_type t) {
    switch (t) {
        case fractal_type::MANDELBROT:   return {-2.0,  1.0, -1.5, 1.5};
        case fractal_type::BURNING_SHIP: return {-2.5,  1.5, -2.0, 1.0};
        case fractal_type::JULIA:        return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::NEWTON:       return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::SIERPINSKI:   return {-0.8,  0.8, -0.05, 0.75};
        case fractal_type::MULTIBROT:    return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::NOVA:         return {-1.5,  1.5, -1.5, 1.5};
        case fractal_type::BARNSLEY:     return {-3.0,  3.0, 10.5, -0.5};
        case fractal_type::LYAPUNOV:     return { 2.0,  4.0,  2.0,  4.0};
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
        case fractal_type::MULTIBROT:    return "multibrot";
        case fractal_type::NOVA:         return "nova";
        case fractal_type::BARNSLEY:     return "barnsley";
        case fractal_type::LYAPUNOV:     return "lyapunov";
    }
    return "mandelbrot";
}

inline bool fractal_type_from_string(const std::string& s, fractal_type& out) {
    if (s == "mandelbrot")   { out = fractal_type::MANDELBROT;   return true; }
    if (s == "burning_ship") { out = fractal_type::BURNING_SHIP; return true; }
    if (s == "julia")        { out = fractal_type::JULIA;        return true; }
    if (s == "newton")       { out = fractal_type::NEWTON;       return true; }
    if (s == "sierpinski")   { out = fractal_type::SIERPINSKI;   return true; }
    if (s == "multibrot")    { out = fractal_type::MULTIBROT;    return true; }
    if (s == "nova")         { out = fractal_type::NOVA;         return true; }
    if (s == "barnsley")     { out = fractal_type::BARNSLEY;     return true; }
    if (s == "lyapunov")     { out = fractal_type::LYAPUNOV;     return true; }
    return false;
}

#endif // FRACTAL_TYPE_H
