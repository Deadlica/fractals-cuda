#ifndef APP_H
#define APP_H

// Project
#include <Fractal/fractal.cuh>
#include <Fractal/palette.cuh>
#include <GUI/coordinate_label.h>
#include <GUI/menu.h>

// SFML
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window/Event.hpp>

// std
#include <atomic>
#include <chrono>
#include <mutex>

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


    void load_fractal(menu::fractal fractal, FractalParams& params);
    void apply_settings(const struct app_settings& s, FractalParams& params);
    void apply_viewport(FractalParams& params, const viewport& v);
    void sync_julia_c(FractalParams& params);
    void start_sierpinski_animation();
    void tick_sierpinski_animation(FractalParams& params, std::atomic<bool>& dirty, std::mutex& mtx);
    void update_texture(sf::Texture& texture, uchar4* h_image);
    bool can_zoom(double x_min, double x_max, double y_min, double y_max, double zoom_factor);
    void compute_fractal(FractalParams& params, std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx);
    void clear_events(sf::RenderWindow& window);

    const std::string WINDOW_NAME = "Fractals";
    static constexpr double MIN_SCALE = 5e-15;
    std::atomic<bool> window_running;
    std::atomic<bool> _compute_paused;
    std::atomic<bool> _compute_idle;

    int _width;
    int _height;
    std::unique_ptr<menu> _menu;
    std::unique_ptr<sf::RenderWindow> _window;
    std::unique_ptr<fractal> _fractal;
    fractal_type _fractal_type;
    uchar4* _h_image;  // current host image buffer (always == params.h_image; see compute_fractal)
    sf::Texture _texture;
    bool _sprite_dirty;
    std::string _pattern;
    std::string _current_theme;

    double _x_min;
    double _x_max;
    double _y_min;
    double _y_max;
    int _max_iter;
    double _zoom_factor;
    bool _smooth;
    double _c_re;
    double _c_im;

    // Sierpinski build-up animation state
    static constexpr int SIERPINSKI_STEP_MS = 250;
    std::chrono::steady_clock::time_point _sierp_anim_start;
    bool _sierp_anim_running;
};


#endif // APP_H
