#ifndef ADD_PATTERN_WINDOW_H
#define ADD_PATTERN_WINDOW_H

// Project
#include <Fractal/fractal_type.h>

// SFML
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Window/VideoMode.hpp>

class add_pattern {
public:
    add_pattern(double x_min, double x_max, double y_min, double y_max,
                int px, int py, const std::string& filename,
                fractal_type type, double c_re, double c_im);
    void run();

private:
    void save_coords();

    int _width;
    int _height;
    sf::RenderWindow _window;
    double _x_min;
    double _x_max;
    double _y_min;
    double _y_max;
    std::string _input_text;
    std::string _file_name;
    fractal_type _type;
    double _c_re;
    double _c_im;
};

#endif // ADD_PATTERN_WINDOW_H
