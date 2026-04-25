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
#include <GUI/options_page.h>
#include <Util/globals.h>

// std
#include <algorithm>
#include <cmath>
#include <thread>

using namespace std::literals::chrono_literals;

static int sierpinski_target_depth(const FractalParams& p);

app::app(int argc, char* argv[], int width, int height):
window_running(true), _compute_paused(false), _compute_idle(true), _width(width), _height(height),
_window(nullptr), _fractal_type(fractal_type::MANDELBROT), _sprite_dirty(false), _pattern(""),
_x_min(-2.0), _x_max(1.0), _y_min(-1.5), _y_max(1.5),
_max_iter(500), _zoom_factor(0.95), _smooth(false),
_c_re(-0.8), _c_im(0.156), _sierp_anim_running(false) {
    std::string theme;
    init_custom_cli_patterns(PATTERNS_PATH);

    bool julia_c_set = false;
    parse_cli_args(argc, argv, _width, _height, _pattern, theme,
                   _max_iter, _zoom_factor, _smooth,
                   _c_re, _c_im, julia_c_set);

    _h_image = new uchar4[_width * _height];

    if (!_pattern.empty()) {
        const goal& g = goals[_pattern];
        _x_min = g.min.x; _x_max = g.max.x;
        _y_min = g.min.y; _y_max = g.max.y;
        _fractal_type = g.type;
        if (g.type == fractal_type::JULIA && !julia_c_set) {
            _c_re = g.c_re; _c_im = g.c_im;
        }
    }

    _menu = std::make_unique<menu>(_width, _height);
    _menu->set_fractal(_fractal_type);

    initialize_palette(theme);
    _current_theme = theme;

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
                            _max_iter, _zoom_factor, _smooth, 1e-10, 12,
                            _c_re, _c_im, _h_image};

    app_settings initial;
    initial.width = _width; initial.height = _height;
    initial.pattern = _pattern; initial.theme = _current_theme;
    initial.max_iter = params.max_iter;
    initial.zoom_factor = params.zoom_factor;
    initial.smooth = params.smooth;
    initial.julia_c_re = _c_re; initial.julia_c_im = _c_im;
    auto on_apply = [&](const app_settings& updated) {
        apply_settings(updated, params);
    };
    _menu->run(_window, params, &initial, on_apply);
    if (_pattern.empty()) {
        // No --pattern given: respect menu selection and start at its default viewport.
        _fractal_type = _menu->selected_fractal();
        apply_viewport(params, default_viewport(_fractal_type));
    }
    load_fractal(_fractal_type, params);
    _fractal->generate(params);

    _texture.create(_width, _height);
    update_texture(_texture, _h_image);
    sf::Sprite sprite(_texture);

    coordinate_label coord_label;
    coord_label.set_position(0, _height);
    coord_label.set_coordinate_string(params.x_min + (params.x_max - params.x_min) / 2.0,
                                      params.y_min + (params.y_max - params.y_min) / 2.0);

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
        tick_sierpinski_animation(params, dirty, mtx);
        if (_sprite_dirty) {
            sprite.setTexture(_texture, true);
            _sprite_dirty = false;
        }
        update_frame(params, _texture, sprite, coord_label, dirty, force_update, frame_counter, frame_skip, mtx);
    }

    window_running = false;
    compute_thread.join();
    delete[] params.h_image;
    free_palette();
    free_device_image_buffer();
}

void app::apply_settings(const app_settings& s, FractalParams& params) {
    _compute_paused = true;
    while (!_compute_idle) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    cudaDeviceSynchronize();

    // Size
    if (s.width != _width || s.height != _height) {
        _width = s.width;
        _height = s.height;
        _window->setSize(sf::Vector2u(_width, _height));
        // keep window centered-ish
        _window->setView(sf::View(sf::FloatRect(0, 0, _width, _height)));
        delete[] _h_image;
        _h_image = new uchar4[_width * _height];
        params.width = _width;
        params.height = _height;
        params.h_image = _h_image;
        _texture.create(_width, _height);
        _sprite_dirty = true;  // sprite needs to be rebound to the new texture
    }

    // Theme
    if (s.theme != _current_theme) {
        _current_theme = s.theme;
        free_palette();
        initialize_palette(_current_theme);
    }

    // Max iter / zoom / smooth
    params.max_iter    = s.max_iter;
    params.zoom_factor = s.zoom_factor;
    params.smooth      = s.smooth;
    _max_iter = s.max_iter;
    _zoom_factor = s.zoom_factor;
    _smooth = s.smooth;

    // Julia c
    _c_re = s.julia_c_re;
    _c_im = s.julia_c_im;
    if (_fractal_type == fractal_type::JULIA) {
        params.c_re = _c_re;
        params.c_im = _c_im;
    }

    // Pattern: applying a pattern sets viewport + fractal + (julia) c.
    if (s.pattern != _pattern) {
        _pattern = s.pattern;
        if (!_pattern.empty()) {
            const goal& g = goals[_pattern];
            _fractal_type = g.type;
            apply_viewport(params, {g.min.x, g.max.x, g.min.y, g.max.y});
            if (g.type == fractal_type::JULIA) {
                _c_re = g.c_re; _c_im = g.c_im;
                params.c_re = _c_re; params.c_im = _c_im;
            }
        }
    }

    _compute_paused = false;
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
            } else if (event.mouseButton.button == sf::Mouse::Right) {
                sf::Event::KeyEvent esc{};
                esc.code = sf::Keyboard::Escape;
                handle_key_press(esc, params, dirty, force_update, mtx);
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
        default:
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
            update_texture(texture, params.h_image);
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
    case sf::Keyboard::Escape: {
        // Build settings snapshot for options page
        app_settings s;
        {
            std::lock_guard<std::mutex> lock(mtx);
            s.width = _width; s.height = _height;
            s.pattern = _pattern; s.theme = _current_theme;
            s.max_iter = params.max_iter;
            s.zoom_factor = params.zoom_factor;
            s.smooth = params.smooth;
            s.julia_c_re = _c_re; s.julia_c_im = _c_im;
        }
        auto on_apply = [&](const app_settings& updated) {
            apply_settings(updated, params);  // pauses compute internally
        };
        _menu->run(_window, params, &s, on_apply);
        fractal_type selected = _menu->selected_fractal();

        std::lock_guard<std::mutex> lock(mtx);
        if (selected != _fractal_type) {
            _fractal_type = selected;
            _pattern.clear();
            apply_viewport(params, default_viewport(_fractal_type));
            sync_julia_c(params);
        }
        load_fractal(_fractal_type, params);
        update_texture(_texture, params.h_image);
        break;
    }
    case sf::Keyboard::S:
        handle_save_pattern(params);
        break;
    case sf::Keyboard::Home:
        reset_view(params, dirty, force_update, mtx);
        break;
    default:
        break;
    }
}

void app::handle_save_pattern(const FractalParams& params) {
    add_pattern add_pattern_box(params.x_min, params.x_max, params.y_min, params.y_max,
                                _window->getPosition().x + _width / 2, _window->getPosition().y + _height / 2,
                                PATTERNS_PATH, _fractal_type, params.c_re, params.c_im);
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
            if (_fractal_type == fractal_type::SIERPINSKI && !_sierp_anim_running) {
                params.depth = sierpinski_target_depth(params);
            }
            dirty = true;
        }
    }
}

void app::reset_view(FractalParams& params, std::atomic<bool>& dirty,
                     std::atomic<bool>& force_update, std::mutex& mtx) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!_pattern.empty()) {
        const goal& g = goals[_pattern];
        if (g.type == _fractal_type) {
            apply_viewport(params, {g.min.x, g.max.x, g.min.y, g.max.y});
            if (_fractal_type == fractal_type::JULIA) {
                params.c_re = g.c_re;
                params.c_im = g.c_im;
            }
            if (_fractal_type == fractal_type::SIERPINSKI) {
                start_sierpinski_animation();
                params.depth = 1;
            }
            dirty = true;
            force_update = true;
            return;
        }
        _pattern.clear();  // stale
    }
    apply_viewport(params, default_viewport(_fractal_type));
    sync_julia_c(params);
    if (_fractal_type == fractal_type::SIERPINSKI) {
        start_sierpinski_animation();
        params.depth = 1;
    }
    dirty = true;
    force_update = true;
}

void app::apply_viewport(FractalParams& params, const viewport& v) {
    params.x_min = v.x_min;
    params.x_max = v.x_max;
    params.y_min = v.y_min;
    params.y_max = v.y_max;
}

void app::sync_julia_c(FractalParams& params) {
    params.c_re = _c_re;
    params.c_im = _c_im;
}

void app::load_fractal(menu::fractal fractal, FractalParams& params) {
    _fractal_type = fractal;
    if (_menu) _menu->set_fractal(fractal);
    switch (fractal) {
    case fractal_type::MANDELBROT:   _fractal = std::make_unique<mandelbrot>();   break;
    case fractal_type::NEWTON:       _fractal = std::make_unique<newton>();       break;
    case fractal_type::BURNING_SHIP: _fractal = std::make_unique<burning_ship>(); break;
    case fractal_type::JULIA:        _fractal = std::make_unique<julia>();        break;
    case fractal_type::SIERPINSKI:   _fractal = std::make_unique<sierpinski>();   break;
    }
    if (fractal == fractal_type::SIERPINSKI) {
        start_sierpinski_animation();
        params.depth = 1;
    }
    _fractal->generate(params);
}

void app::start_sierpinski_animation() {
    _sierp_anim_start = std::chrono::steady_clock::now();
    _sierp_anim_running = true;
}

// Resolution-matching depth: enough levels so each fractal cell is ~<=1 pixel.
// Grows with zoom (smaller viewport -> larger depth). Capped at 50 to stay
// within double precision.
static int sierpinski_target_depth(const FractalParams& p) {
    double span = std::max(p.x_max - p.x_min, p.y_max - p.y_min);
    if (span <= 0.0) return 1;
    double pixels = static_cast<double>(std::max(p.width, p.height));
    int d = static_cast<int>(std::ceil(std::log2(pixels / span)));
    if (d < 1)  d = 1;
    if (d > 50) d = 50;
    return d;
}

// Called from the main thread without the mutex held. Takes the lock only to
// update shared state.
void app::tick_sierpinski_animation(FractalParams& params, std::atomic<bool>& dirty, std::mutex& mtx) {
    if (!_sierp_anim_running || _fractal_type != fractal_type::SIERPINSKI) return;

    int target_cap = sierpinski_target_depth(params);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - _sierp_anim_start).count();
    int target = 1 + static_cast<int>(elapsed / SIERPINSKI_STEP_MS);
    if (target > target_cap) target = target_cap;

    std::lock_guard<std::mutex> lock(mtx);
    if (target != params.depth) {
        params.depth = target;
        dirty = true;
    }
    if (target >= target_cap) _sierp_anim_running = false;
}

void app::update_texture(sf::Texture& texture, uchar4* h_image) {
    texture.update(reinterpret_cast<const sf::Uint8*>(h_image));
}

bool app::can_zoom(double x_min, double x_max, double y_min, double y_max, double zoom_factor) {
    double new_width = (x_max - x_min) * zoom_factor;
    double new_height = (y_max - y_min) * zoom_factor;
    return std::abs(new_width) >= MIN_SCALE && std::abs(new_height) >= MIN_SCALE;
}

void app::compute_fractal(FractalParams& params, std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx) {
    while (window_running) {
        if (_compute_paused) {
            _compute_idle = true;
            std::this_thread::sleep_for(10ms);
            continue;
        }
        if (dirty) {
            _compute_idle = false;
            FractalParams temp_params;
            {
                std::lock_guard<std::mutex> lock(mtx);
                temp_params = params;
            }
            temp_params.h_image = new uchar4[temp_params.width * temp_params.height];
            _fractal->generate(temp_params);

            {
                std::lock_guard<std::mutex> lock(mtx);
                std::swap(params.h_image, temp_params.h_image);
                delete[] temp_params.h_image;
                _h_image = params.h_image;  // keep app's tracked pointer in sync
                dirty = false;
                force_update = false;
            }
        }
        _compute_idle = true;
        std::this_thread::sleep_for(10ms);
    }
}

void app::clear_events(sf::RenderWindow& window) {
    sf::Event event;
    while (window.pollEvent(event)) {}
}
