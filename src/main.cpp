#include <CLI/cli.h>
#include <GUI/add_pattern.h>
#include <Fractal/mandelbrot.cuh>
#include <GUI/coordinate_label.h>
#include <GUI/menu.h>
#include <chrono>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <mutex>
#include <atomic>
#include <thread>
#include <sstream>
#include <iomanip>

const std::string WINDOW_NAME = "Mandelbrot Set";
const std::string custom_patterns_file = ".patterns.txt";
constexpr double MIN_SCALE = 5e-15;
std::atomic<bool> window_running(true);

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

void update_texture(sf::Texture& texture, Color* h_image, int width, int height) {
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

bool can_zoom(double x_min, double x_max, double y_min, double y_max, double zoom_factor) {
    double new_width = (x_max - x_min) * zoom_factor;
    double new_height = (y_max - y_min) * zoom_factor;
    return std::abs(new_width) >= MIN_SCALE && std::abs(new_height) >= MIN_SCALE;
}

void compute_mandelbrot(FractalParams& params, std::atomic<bool>& dirty, std::atomic<bool>& force_update, std::mutex& mtx) {
    while (window_running) {
        if (dirty) {
            Color* temp_image = new Color[params.width * params.height];
            mandelbrot(temp_image, params.width, params.height, params.x_min, params.x_max, params.y_min, params.y_max,
                       params.max_iter, params.smooth);

            {
                std::lock_guard<std::mutex> lock(mtx);
                std::swap(params.h_image, temp_image);
                delete[] temp_image;
                dirty = false;
                force_update = false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void clear_events(sf::RenderWindow& window) {
    sf::Event event;
    while (window.pollEvent(event)) {}
}

int main(int argc, char* argv[]) {
    int width = 800;
    int height = 600;
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    std::string pattern;
    std::string theme;
    int max_iter = 500;
    double zoom_factor = 0.95;
    bool smooth = false;
    init_custom_cli_patterns(custom_patterns_file);
    parse_cli_args(argc, argv, width, height, pattern, theme, max_iter, zoom_factor, smooth);
    Color *h_image = new Color[width * height];

    if (!pattern.empty()) {
        goal g = goals[pattern];
        x_min = g.min.x;
        x_max = g.max.x;
        y_min = g.min.y;
        y_max = g.max.y;
    }

    initialize_palette(theme);
    mandelbrot(h_image, width, height, x_min, x_max, y_min, y_max, max_iter, smooth);

    sf::RenderWindow window(sf::VideoMode(width, height), WINDOW_NAME, sf::Style::Titlebar | sf::Style::Close);
    int monitors = sf::VideoMode::getFullscreenModes().size();
    int x_offset = monitors / 2 * 1920;

    window.setPosition(sf::Vector2i(
        sf::VideoMode::getDesktopMode().width * 0.5 - window.getSize().x * 0.5 + x_offset,
        sf::VideoMode::getDesktopMode().height * 0.5 - window.getSize().y * 0.5)
    );

    menu menu(width, height);
    menu.run(window);

    std::ostringstream oss;
    oss << std::setprecision(16);

    sf::Texture texture;
    texture.create(width, height);
    update_texture(texture, h_image, width, height);
    sf::Sprite sprite(texture);

    coordinate_label coord_label;
    coord_label.set_position(0, height);
    coord_label.set_coordinate_string(x_min + (x_max - x_min) / 2.0, y_min + (y_max - y_min) / 2.0);

    FractalParams params = {width, height, x_min, x_max, y_min, y_max,
                            max_iter, zoom_factor, smooth, h_image};

    std::atomic<bool> dirty(false);
    std::atomic<bool> force_update(false);
    std::mutex mtx;
    std::thread compute_thread(compute_mandelbrot, std::ref(params), std::ref(dirty), std::ref(force_update), std::ref(mtx));

    bool is_dragging = false;
    sf::Vector2i prev_mouse_pos;
    const int drag_delay_ms = 10;
    const int frame_skip = 2;
    int frame_counter = 0;

    auto last_update = std::chrono::steady_clock::now();

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::KeyPressed) {
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    window.close();
                    break;
                case sf::Keyboard::S:
                {
                    add_pattern add_pattern_box(params.x_min, params.x_max, params.y_min, params.y_max,
                                                window.getPosition().x + width / 2, window.getPosition().y + height / 2,
                                                custom_patterns_file);
                    add_pattern_box.run();
                    clear_events(window);
                    window.setActive();
                    break;
                }
                case sf::Keyboard::Home:
                    {
                        std::lock_guard<std::mutex> lock(mtx);
                        if (!pattern.empty()) {
                            goal g = goals[pattern];
                            params.x_min = g.min.x;
                            params.x_max = g.max.x;
                            params.y_min = g.min.y;
                            params.y_max = g.max.y;
                        }
                        else {
                            params.x_min = -2.0;
                            params.x_max = 1.0;
                            params.y_min = -1.5;
                            params.y_max = 1.5;
                        }
                        dirty = true;
                        force_update = true;
                    }
                    frame_counter = frame_skip - 1;
                    break;
                }
            }
	        else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
		        is_dragging = true;
		        prev_mouse_pos = sf::Mouse::getPosition(window);
    	    	last_update = std::chrono::steady_clock::now();
	        }
	        else if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Left) {
		        is_dragging = false;
	        }
	        else if (event.type == sf::Event::MouseMoved && is_dragging) {
    	    	auto now = std::chrono::steady_clock::now();
		        auto time_passed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
		        if (time_passed >= drag_delay_ms) {
		            sf::Vector2i curr_mouse_pos = sf::Mouse::getPosition(window);
                    sf::Vector2i delta = curr_mouse_pos - prev_mouse_pos;

                    double dx = (params.x_max - params.x_min) * delta.x / window.getSize().x;
                    double dy = (params.y_max - params.y_min) * delta.y / window.getSize().y;

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
            else if (event.type == sf::Event::MouseWheelScrolled) {
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);

                double x_center_before = params.x_min + mouse_pos.x * (params.x_max - params.x_min) / params.width;
                double y_center_before = params.y_min + mouse_pos.y * (params.y_max - params.y_min) / params.height;

                zoom_factor = (event.mouseWheelScroll.delta > 0) ? params.zoom_factor : 1.0 / params.zoom_factor;

                if (can_zoom(params.x_min, params.x_max, params.y_min, params.y_max, zoom_factor)) {
                    double new_width = (params.x_max - params.x_min) * zoom_factor;
                    double new_height = (params.y_max - params.y_min) * zoom_factor;
                    
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
            else if (event.type == sf::Event::Resized) {
                window.setSize(sf::Vector2<unsigned int>(width, height));
            }
        }

        if ((++frame_counter % frame_skip) == 0) {
            frame_counter = 0;
            if (dirty) {
                while (force_update) continue;
                std::lock_guard<std::mutex> lock(mtx);
                update_texture(texture, params.h_image, params.width, params.height);
                coord_label.set_coordinate_string(params.x_min + (params.x_max - params.x_min) / (double)2.0,
                                                  params.y_min + (params.y_max - params.y_min) / (double)2.0);
            }

            window.clear();
            window.draw(sprite);
            window.draw(coord_label);
            window.display();
        }
    }

    window_running = false;
    compute_thread.join();
    delete[] params.h_image;
    free_palette();
    return 0;
}
