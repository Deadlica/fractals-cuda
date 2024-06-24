#include "cli.h"
#include "mandelbrot.h"
#include <chrono>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Window/VideoMode.hpp>

const std::string WINDOW_NAME = "Mandelbrot Set";

struct MandelbrotParams {
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

int main(int argc, char* argv[]) {
    int width = 600;
    int height = 600;
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    std::string pattern = "";
    std::string theme = "";
    int max_iter = 1500;
    double zoom_factor = 0.95;
    bool smooth = false;
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

    sf::RenderWindow window(sf::VideoMode(width, height), WINDOW_NAME);
    int monitors = sf::VideoMode::getFullscreenModes().size();
    int x_offset = monitors / 2 * 1920;

    window.setPosition(sf::Vector2i(
        sf::VideoMode::getDesktopMode().width * 0.5 - window.getSize().x * 0.5 + x_offset,
        sf::VideoMode::getDesktopMode().height * 0.5 - window.getSize().y * 0.5)
    );

    sf::Texture texture;
    texture.create(width, height);
    update_texture(texture, h_image, width, height);
    sf::Sprite sprite(texture);

    MandelbrotParams params = {width, height, x_min, x_max, y_min, y_max,
                               max_iter, zoom_factor, smooth, h_image};

    bool is_dragging = false;
    sf::Vector2i prev_mouse_pos;
    const int drag_delay_ms = 10;

    auto last_update = std::chrono::steady_clock::now();

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed ||
                event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) {
                window.close();
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

                    params.x_min -= dx;
                    params.x_max -= dx;
                    params.y_min -= dy;
                    params.y_max -= dy;

		            mandelbrot(params.h_image, params.width, params.height, params.x_min, params.x_max,
                               params.y_min, params.y_max, params.max_iter, params.smooth);
                    update_texture(texture, params.h_image, params.width, params.height);

                    prev_mouse_pos = curr_mouse_pos;
		            last_update = now;
                }
	        }
            else if (event.type == sf::Event::MouseWheelScrolled) {
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);

                double x_center_before = params.x_min + mouse_pos.x * (params.x_max - params.x_min) / params.width;
                double y_center_before = params.y_min + mouse_pos.y * (params.y_max - params.y_min) / params.height;

                double zoom_factor = (event.mouseWheelScroll.delta > 0) ? params.zoom_factor : 1.0 / params.zoom_factor;

                double new_width = (params.x_max - params.x_min) * zoom_factor;
                double new_height = (params.y_max - params.y_min) * zoom_factor;
                params.x_min = x_center_before - (mouse_pos.x / (double) params.width) * new_width;
                params.x_max = x_center_before + (1 - mouse_pos.x / (double) params.width) * new_width;
                params.y_min = y_center_before - (mouse_pos.y / (double) params.height) * new_height;
                params.y_max = y_center_before + (1 - mouse_pos.y / (double) params.height) * new_height;

                mandelbrot(params.h_image, params.width, params.height, params.x_min, params.x_max,
                           params.y_min, params.y_max, params.max_iter, params.smooth);
                update_texture(texture, params.h_image, params.width, params.height);
            }
        }
        window.clear();
        window.draw(sprite);
        window.display();
    }

    delete[] h_image;
    free_palette();
    return 0;
}
