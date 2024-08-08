// Project
#include <CLI/cli.h>
#include <Fractal/burning_ship.cuh>
#include <Fractal/julia.cuh>
#include <Fractal/mandelbrot.cuh>
#include <Fractal/newton.cuh>
#include <Fractal/sierpinski.cuh>
#include <GUI/app.h>
#include <GUI/menu.h>
#include <GUI/add_pattern.h>
#include <Util/globals.h>

// std
#include <thread>

using namespace std::literals::chrono_literals;

app::app(int argc, char* argv[], int width, int height):
window_running(true), _width(width), _height(height), _menu(_width, _height),
_window(nullptr), _pattern(""), _x_min(-2.0), _x_max(1.0),
_y_min(-1.5), _y_max(1.5), _max_iter(500), _zoom_factor(0.95), _smooth(false) {
    std::string theme;
    init_custom_cli_patterns(PATTERNS_PATH);
    parse_cli_args(argc, argv, width, height, _pattern, theme, _max_iter, _zoom_factor, _smooth);
    _h_image = new Color[width * height];

    if (!_pattern.empty()) {
        goal g = goals[_pattern];
        _x_min = g.min.x;
        _x_max = g.max.x;
        _y_min = g.min.y;
        _y_max = g.max.y;
    }

    initialize_palette(theme);

    _window = std::make_unique<sf::RenderWindow>(sf::VideoMode(_width, _height), WINDOW_NAME, sf::Style::Titlebar | sf::Style::Close);
    int monitors = sf::VideoMode::getFullscreenModes().size();
    int x_offset = monitors / 2 * 1920;

    _window->setPosition(sf::Vector2i(
            sf::VideoMode::getDesktopMode().width * 0.5 - _window->getSize().x * 0.5 + x_offset,
            sf::VideoMode::getDesktopMode().height * 0.5 - _window->getSize().y * 0.5)
    );
}

void app::run() {
    FractalParams params = {_width, _height, _x_min, _x_max, _y_min, _y_max,
                            _max_iter, _zoom_factor, _smooth, 1e-10, 100, _h_image};

    _menu.run(_window, params);
    load_fractal(_menu.selected_fractal(), params);

    _fractal->generate(params);

    sf::Texture texture;
    texture.create(_width, _height);
    update_texture(texture, _h_image, _width, _height);
    sf::Sprite sprite(texture);

    coordinate_label coord_label;
    coord_label.set_position(0, _height);
    coord_label.set_coordinate_string(_x_min + (_x_max - _x_min) / 2.0, _y_min + (_y_max - _y_min) / 2.0);



    std::atomic<bool> dirty(false);
    std::atomic<bool> force_update(false);
    std::mutex mtx;
    std::thread compute_thread(&app::compute_fractal, this, std::ref(params), std::ref(dirty), std::ref(force_update), std::ref(mtx));

    bool is_dragging = false;
    sf::Vector2i prev_mouse_pos;
    const int drag_delay_ms = 10;
    const int frame_skip = 2;
    int frame_counter = 0;

    auto last_update = std::chrono::steady_clock::now();

    while (_window->isOpen()) {
        handle_events(params, is_dragging, prev_mouse_pos, dirty, force_update, mtx, drag_delay_ms, last_update);
        update_frame(params, texture, sprite, coord_label, dirty, force_update, frame_counter, frame_skip, mtx);
    }

    window_running = false;
    compute_thread.join();
    delete[] params.h_image;
    free_palette();
}

void app::handle_events(FractalParams& params, bool& is_dragging, sf::Vector2i& prev_mouse_pos,
                        std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx,
                        const int drag_delay_ms, std::chrono::steady_clock::time_point& last_update) {
    sf::Event event;
    while (_window->pollEvent(event)) {
        switch (event.type) {
        case sf::Event::Closed:
            _window->close();
            break;
        case sf::Event::KeyPressed:
            handle_key_press(event.key, params, dirty, force_update, mtx);
            break;
        case sf::Event::MouseButtonPressed:
            if (event.mouseButton.button == sf::Mouse::Left) {
                is_dragging = true;
                prev_mouse_pos = sf::Mouse::getPosition(*_window);
            }
            break;
        case sf::Event::MouseButtonReleased:
            if (event.mouseButton.button == sf::Mouse::Left) {
                is_dragging = false;
            }
            break;
        case sf::Event::MouseMoved:
            if (is_dragging) {
                handle_mouse_drag(params, prev_mouse_pos, dirty, mtx, drag_delay_ms, last_update);
            }
            break;
        case sf::Event::MouseWheelScrolled:
            handle_mouse_scroll(event.mouseWheelScroll, params, dirty, mtx);
            break;
        case sf::Event::Resized:
            _window->setSize(sf::Vector2<unsigned int>(_width, _height));
            break;
        }
    }
}

void app::update_frame(FractalParams& params, sf::Texture& texture, sf::Sprite& sprite,
                       coordinate_label& coord_label, std::atomic<bool>& dirty,
                       std::atomic<bool>& force_update, int& frame_counter, const int frame_skip, std::mutex& mtx) {
    if ((++frame_counter % frame_skip) == 0) {
        frame_counter = 0;
        if (dirty) {
            while (force_update) continue;
            std::lock_guard<std::mutex> lock(mtx);
            update_texture(texture, params.h_image, params.width, params.height);
            coord_label.set_coordinate_string(params.x_min + (params.x_max - params.x_min) / 2.0,
                                              params.y_min + (params.y_max - params.y_min) / 2.0);
        }

        _window->clear();
        _window->draw(sprite);
        _window->draw(coord_label);
        _window->display();
    }
}

void app::handle_key_press(const sf::Event::KeyEvent& key, FractalParams& params,
                           std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx) {
    switch (key.code) {
    case sf::Keyboard::Escape:
        _menu.run(_window, params);
        {
            std::lock_guard<std::mutex> lock(mtx);
            load_fractal(_menu.selected_fractal(), params);
            dirty = true;
        }
        break;
    case sf::Keyboard::S:
        handle_save_pattern(params);
        break;
    case sf::Keyboard::Home:
        reset_view(params, dirty, force_update, mtx);
        break;
    }
}

void app::handle_save_pattern(const FractalParams& params) {
    add_pattern add_pattern_box(params.x_min, params.x_max, params.y_min, params.y_max,
                                _window->getPosition().x + _width / 2, _window->getPosition().y + _height / 2,
                                PATTERNS_PATH);
    add_pattern_box.run();
    clear_events(*_window);
    _window->setActive();
}

void app::handle_mouse_drag(FractalParams& params, sf::Vector2i& prev_mouse_pos, std::atomic<bool>& dirty,
                            std::mutex& mtx, const int drag_delay_ms, std::chrono::steady_clock::time_point& last_update) {
    auto now = std::chrono::steady_clock::now();
    auto time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
    if (time_passed >= drag_delay_ms) {
        sf::Vector2i curr_mouse_pos = sf::Mouse::getPosition(*_window);
        sf::Vector2i delta = curr_mouse_pos - prev_mouse_pos;

        double dx = (params.x_max - params.x_min) * delta.x / _window->getSize().x;
        double dy = (params.y_max - params.y_min) * delta.y / _window->getSize().y;

        {
            std::lock_guard<std::mutex> lock(mtx);
            params.x_min -= dx;
            params.x_max -= dx;
            params.y_min -= dy;
            params.y_max -= dy;
            dirty = true;
        }

        prev_mouse_pos = curr_mouse_pos;
        last_update = now;
    }
}

void app::handle_mouse_scroll(const sf::Event::MouseWheelScrollEvent& scroll, FractalParams& params,
                              std::atomic<bool>& dirty, std::mutex& mtx) {
    sf::Vector2i mouse_pos = sf::Mouse::getPosition(*_window);

    double x_center_before = params.x_min + mouse_pos.x * (params.x_max - params.x_min) / params.width;
    double y_center_before = params.y_min + mouse_pos.y * (params.y_max - params.y_min) / params.height;

    _zoom_factor = (scroll.delta > 0) ? params.zoom_factor : 1.0 / params.zoom_factor;

    if (can_zoom(params.x_min, params.x_max, params.y_min, params.y_max, _zoom_factor)) {
        double new_width = (params.x_max - params.x_min) * _zoom_factor;
        double new_height = (params.y_max - params.y_min) * _zoom_factor;

        {
            std::lock_guard<std::mutex> lock(mtx);
            params.x_min = x_center_before - (mouse_pos.x / (double) params.width) * new_width;
            params.x_max = x_center_before + (1 - mouse_pos.x / (double) params.width) * new_width;
            params.y_min = y_center_before - (mouse_pos.y / (double) params.height) * new_height;
            params.y_max = y_center_before + (1 - mouse_pos.y / (double) params.height) * new_height;
            dirty = true;
        }
    }
}

void app::reset_view(FractalParams& params, std::atomic<bool>& dirty,
                     std::atomic<bool>& force_update, std::mutex& mtx) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!_pattern.empty()) {
        goal g = goals[_pattern];
        params.x_min = g.min.x;
        params.x_max = g.max.x;
        params.y_min = g.min.y;
        params.y_max = g.max.y;
    } else {
        params.x_min = -2.0;
        params.x_max = 1.0;
        params.y_min = -1.5;
        params.y_max = 1.5;
    }
    dirty = true;
    force_update = true;
}

void app::load_fractal(menu::fractal fractal, const FractalParams& params) {
    switch (fractal) {
    case menu::fractal::MANDELBROT:
        _fractal = std::make_unique<mandelbrot>();
        break;
    case menu::fractal::NEWTON:
        _fractal = std::make_unique<newton>();
        break;
    case menu::fractal::BURNING_SHIP:
        _fractal = std::make_unique<burning_ship>();
        break;
    case menu::fractal::JULIA:
        _fractal = std::make_unique<julia>();
        break;
    case menu::fractal::SIERPINSKI:
        _fractal = std::make_unique<sierpinski>();
        break;
    }
    _fractal->generate(params);
}

void app::update_texture(sf::Texture& texture, Color* h_image, int width, int height) {
    sf::Uint8* pixels = new sf::Uint8[width * height * 4];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Color color = h_image[y * width + x];
            int index = (y * width + x) * 4;
            pixels[index + 0] = color.r;
            pixels[index + 1] = color.g;
            pixels[index + 2] = color.b;
            pixels[index + 3] = 255;
        }
    }
    texture.update(pixels);
    delete[] pixels;
}

bool app::can_zoom(double x_min, double x_max, double y_min, double y_max, double zoom_factor) {
    double new_width = (x_max - x_min) * zoom_factor;
    double new_height = (y_max - y_min) * zoom_factor;
    return std::abs(new_width) >= MIN_SCALE && std::abs(new_height) >= MIN_SCALE;
}

void app::compute_fractal(FractalParams& params, std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx) {
    while (window_running) {
        if (dirty) {
            FractalParams temp_params = params;
            temp_params.h_image = new Color[params.width * params.height];
            _fractal->generate(temp_params);

            {
                std::lock_guard<std::mutex> lock(mtx);
                std::swap(params.h_image, temp_params.h_image);
                delete[] temp_params.h_image;
                dirty = false;
                force_update = false;
            }
        }
        std::this_thread::sleep_for(10ms);
    }
}

void app::clear_events(sf::RenderWindow& window) {
    sf::Event event;
    while (window.pollEvent(event)) {}
}