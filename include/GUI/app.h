#ifndef APP_H
#define APP_H

// Project
#include <Fractal/palette.cuh>
#include <GUI/coordinate_label.h>

// SFML
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window/Event.hpp>

// std
#include <atomic>
#include <mutex>

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
    void handle_events(FractalParams& params, bool& is_dragging, sf::Vector2i& prev_mouse_pos,
                       std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx,
                       const int drag_delay_ms, std::chrono::steady_clock::time_point& last_update);
    void update_frame(FractalParams& params, sf::Texture& texture, sf::Sprite& sprite,
                      coordinate_label& coord_label, std::atomic<bool>& dirty,
                      std::atomic<bool>& force_update, int& frame_counter, const int frame_skip, std::mutex& mtx);
    void handle_key_press(const sf::Event::KeyEvent& key, FractalParams& params,
                          std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx);
    void handle_save_pattern(const FractalParams& params);
    void handle_mouse_drag(FractalParams& params, sf::Vector2i& prev_mouse_pos, std::atomic<bool>& dirty,
                           std::mutex& mtx, const int drag_delay_ms, std::chrono::steady_clock::time_point& last_update);
    void handle_mouse_scroll(const sf::Event::MouseWheelScrollEvent& scroll, FractalParams& params,
                             std::atomic<bool>& dirty, std::mutex& mtx);
    void reset_view(FractalParams& params, std::atomic<bool>& dirty,
                    std::atomic<bool>& force_update, std::mutex& mtx);


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
