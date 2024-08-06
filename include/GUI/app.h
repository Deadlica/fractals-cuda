#ifndef APP_H
#define APP_H

#include <Fractal/palette.cuh>
#include <SFML/Graphics/Texture.hpp>
#include <atomic>
#include <mutex>
#include <SFML/Graphics/RenderWindow.hpp>

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
public:
    app(int argc, char* argv[], int width, int height);

    void run();

private:
    void update_texture(sf::Texture& texture, Color* h_image, int width, int height);
    bool can_zoom(double x_min, double x_max, double y_min, double y_max, double zoom_factor);
    void compute_mandelbrot(FractalParams& params, std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx);
    void clear_events(sf::RenderWindow& window);

    const std::string WINDOW_NAME = "Mandelbrot Set";
    const std::string custom_patterns_file = ".patterns.txt";
    static constexpr double MIN_SCALE = 5e-15;
    std::atomic<bool> window_running;

    int _width;
    int _height;
    std::unique_ptr<sf::RenderWindow> _window;
    Color* _h_image;
    std::string _pattern;

    double _x_min;
    double _x_max;
    double _y_min;
    double _y_max;
    int _max_iter;
    double _zoom_factor;
    bool _smooth;
};


#endif // APP_H
