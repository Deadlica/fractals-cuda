// Project
#include <GUI/menu.h>
#include <Util/globals.h>

// std
#include <cmath>
#include <iostream>
#include <valarray>

menu::menu(float width, float height):
_current_fractal(menu::fractal::MANDELBROT), _current_mode(mode::MAIN),
_is_closing(false), _background(animation::Type::SQUARES) {
    if (!_font.loadFromFile(MENU_FONT_PATH)) {
        std::cout << "Failed to load font!\nEnsure \"" + MENU_FONT_PATH + "\" exists" << std::endl;
        exit(0);
    }
    _box_color = sf::Color(74, 74, 74);
    _text_color = sf::Color(255, 255, 255);
    _selected_text_color = sf::Color(255, 69, 0);
    _selected_box_color = sf::Color(64, 64, 64);
    _begin[mode::MAIN] = main_buttons::START;
    _begin[mode::FRACTALS] = fractal_buttons::MANDELBROT;
    _begin[mode::OPTIONS] = option_buttons::SIZE;
    _end[mode::MAIN] = main_buttons::EXIT;
    _end[mode::FRACTALS] = fractal_buttons::SIERPINSKI;
    _end[mode::OPTIONS] = option_buttons::MAX_ITER;

    init_menu(width, height);
}

menu::~menu() {}

void menu::run(std::unique_ptr<sf::RenderWindow>& window, FractalParams& params) {
    _is_closing = false;
    _window = window.get();
    _background.initialize(*_window);

    while (_window->isOpen()) {
        sf::Event event;
        while (_window->pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                _window->close();
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    _window->close();
                    return;
                case sf::Keyboard::Up:
                    move_up();
                    break;
                case sf::Keyboard::Down:
                    move_down();
                    break;
                case sf::Keyboard::Return:
                    action(_selected_index[_current_mode]);
                    _current_mode = mode::MAIN;
                    break;
                }
                break;
            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    handle_mouse_click();
                    _current_mode = mode::MAIN;
                }
                break;
            }
        }

        if (_is_closing) {
            if (_selected_index[mode::MAIN] == main_buttons::EXIT) {
                _window->close();
            }
            return;
        }

        update_hover();
        _window->clear();
        draw();
        _window->display();
    }
}

menu::fractal menu::selected_fractal() const {
    return _current_fractal;
}

void menu::init_menu(float width, float height) {
    std::vector<std::vector<std::string>> menus = {
        {"Start", "Fractals", "Options", "Exit" },
        {"Mandelbrot", "Newton", "Burning Ship", "Julia", "Sierpinski" }
    };
    std::vector<mode> modes = { mode::MAIN, mode::FRACTALS };
    float box_width = width * 0.3;
    float box_height = height * 0.1;

    size_t i = 0;
    for (const auto& options : menus) {
        mode current_mode = modes[i];
        for (size_t j = 0; j < options.size(); ++j) {
            sf::Text text;
            text.setFont(_font);
            text.setString(options[j]);
            text.setFillColor(j == 0 ? _selected_text_color : _text_color);
            text.setCharacterSize(30);

            sf::RectangleShape box(sf::Vector2f(box_width, box_height));
            box.setFillColor(j == 0 ? _selected_box_color : _box_color);
            box.setOrigin(box.getSize().x / 2, box.getSize().y / 2);
            box.setPosition(sf::Vector2f(width / 2, height / (options.size() + 1) * (j + 1)));

            sf::FloatRect text_bounds = text.getLocalBounds();
            text.setOrigin(text_bounds.left + text_bounds.width / 2.0f, text_bounds.top + text_bounds.height / 2.0f);
            text.setPosition(box.getPosition());

            _menu_boxes[current_mode].push_back(box);
            _menu_texts[current_mode].push_back(text);
        }
        i++;
    }

    _selected_index[_current_mode] = main_buttons::START;
}

#include <iostream>
void menu::draw() {
    _background.update(*_window);
    _window->draw(_background);
    for (const auto& box : _menu_boxes[_current_mode]) {
        _window->draw(box);
    }
    for (const auto& text : _menu_texts[_current_mode]) {
        _window->draw(text);
    }
}

void menu::move_up() {
    if (_selected_index[_current_mode] > _begin[_current_mode]) {
        move_to_button(_selected_index[_current_mode] - 1);
    }
}

void menu::move_down() {
    if (_selected_index[_current_mode] < _end[_current_mode]) {
        move_to_button(_selected_index[_current_mode] + 1);
    }
}

void menu::handle_mouse_click() {
    sf::Vector2i mouse_pos = sf::Mouse::getPosition(*_window);
    sf::Vector2f world_pos = _window->mapPixelToCoords(mouse_pos);

    for (size_t i = 0; i < _menu_boxes[_current_mode].size(); ++i) {
        sf::FloatRect bounds = _menu_boxes[_current_mode][i].getGlobalBounds();
        if (bounds.contains(world_pos)) {
            _selected_index[_current_mode] = i;
            action(_selected_index[_current_mode]);
            break;
        }
    }
}

void menu::action(int index) {
    switch (_current_mode) {
    case mode::MAIN:
        switch (index) {
        case main_buttons::START:
            _is_closing = true;
            break;
        case main_buttons::FRACTALS:
            load_fractals_menu();
            break;
        case main_buttons::OPTIONS:
            load_options_menu();
            break;
        case main_buttons::EXIT:
            _is_closing = true;
            break;
        }
        break;
    case mode::FRACTALS:
        switch (index) {
        case fractal_buttons::MANDELBROT:
            _current_fractal = fractal::MANDELBROT;
            break;
        case fractal_buttons::NEWTON:
            _current_fractal = fractal::NEWTON;
            break;
        case fractal_buttons::BURNING_SHIP:
            _current_fractal = fractal::BURNING_SHIP;
            break;
        case fractal_buttons::JULIA:
            _current_fractal = fractal::JULIA;
            break;
        case fractal_buttons::SIERPINSKI:
            _current_fractal = fractal::SIERPINSKI;
            break;
        }
        break;
    case mode::OPTIONS:
        break;
    }
}

void menu::load_fractals_menu() {
    _current_mode = mode::FRACTALS;
    int old_index = _selected_index[_current_mode];
    while (_window->isOpen()) {
        sf::Event event;
        while (_window->pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                _window->close();
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                case sf::Keyboard::Escape:
                    move_to_button(old_index);
                    return;
                case sf::Keyboard::Up:
                    move_up();
                    break;
                case sf::Keyboard::Down:
                    move_down();
                    break;
                case sf::Keyboard::Return:
                    action(_selected_index[_current_mode]);
                    return;
                }
                break;
            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    handle_mouse_click();
                    return;
                }
                break;
            }
        }

        update_hover();
        _window->clear();
        draw();
        _window->display();
    }
}

void menu::load_options_menu() {
    _current_mode = mode::OPTIONS;
}

void menu::move_to_button(int index) {
    _menu_texts[_current_mode][_selected_index[_current_mode]].setFillColor(_text_color);
    _menu_boxes[_current_mode][_selected_index[_current_mode]].setFillColor(_box_color);
    _selected_index[_current_mode] = index;
    _menu_texts[_current_mode][_selected_index[_current_mode]].setFillColor(_selected_text_color);
    _menu_boxes[_current_mode][_selected_index[_current_mode]].setFillColor(_selected_box_color);
}

void menu::update_hover() {
    sf::Vector2i mouse_pos = sf::Mouse::getPosition(*_window);
    sf::Vector2f world_pos = _window->mapPixelToCoords(mouse_pos);

    bool hover_detected = false;

    for (size_t i = 0; i < _menu_boxes[_current_mode].size(); ++i) {
        sf::FloatRect bounds = _menu_boxes[_current_mode][i].getGlobalBounds();
        if (bounds.contains(world_pos)) {
            if (_selected_index[_current_mode] != i) {
                move_to_button(i);
            }
            hover_detected = true;
            break;
        }
    }

    if (!hover_detected && _selected_index[_current_mode] >= _begin[_current_mode] && _selected_index[_current_mode] < _menu_texts[_current_mode].size()) {
        _menu_texts[_current_mode][_selected_index[_current_mode]].setFillColor(_selected_text_color);
        _menu_boxes[_current_mode][_selected_index[_current_mode]].setFillColor(_selected_box_color);
    }
}