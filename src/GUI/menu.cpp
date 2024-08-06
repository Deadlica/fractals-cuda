// Project
#include <GUI/menu.h>

// std
#include <iostream>

menu::menu(float width, float height) {
    if (!_font.loadFromFile("fonts/arial.ttf")) {
        std::cout << "Failed to load font!\nEnsure \"arial.ttf\" is located in fonts/" << std::endl;
        exit(0);
    }

    init_menu(width, height);
}

menu::~menu() {}

void menu::run(std::unique_ptr<sf::RenderWindow>& window) {
    while (window->isOpen()) {
        sf::Event event;
        while (window->pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window->close();
            } else if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Up) {
                    move_up();
                } else if (event.key.code == sf::Keyboard::Down) {
                    move_down();
                } else if (event.key.code == sf::Keyboard::Return) {
                    if (_selected_index == 0) {
                        return;
                    } else if (_selected_index == 1) {
                        // Options selected
                    } else if (_selected_index == 2) {
                        window->close();
                    }
                }
            } else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    handle_mouse_click(window);
                }
            }
        }

        update_hover(window);

        window->clear();
        draw(window);
        window->display();
    }
}

void menu::init_menu(float width, float height) {
    std::vector<std::string> options = {"Play", "Options", "Exit"};

    for (size_t i = 0; i < options.size(); ++i) {
        sf::Text text;
        text.setFont(_font);
        text.setString(options[i]);
        text.setFillColor(i == 0 ? sf::Color::Red : sf::Color::White);
        text.setPosition(sf::Vector2f(width / 2, height / (options.size() + 1) * (i + 1)));
        _menu_options.push_back(text);
    }

    _selected_index = 0;
}

void menu::draw(std::unique_ptr<sf::RenderWindow> &window) {
    for (const auto &option : _menu_options) {
        window->draw(option);
    }
}

void menu::move_up() {
    if (_selected_index - 1 >= 0) {
        _menu_options[_selected_index].setFillColor(sf::Color::White);
        _selected_index--;
        _menu_options[_selected_index].setFillColor(sf::Color::Red);
    }
}

void menu::move_down() {
    if (_selected_index + 1 < _menu_options.size()) {
        _menu_options[_selected_index].setFillColor(sf::Color::White);
        _selected_index++;
        _menu_options[_selected_index].setFillColor(sf::Color::Red);
    }
}

void menu::handle_mouse_click(std::unique_ptr<sf::RenderWindow>& window) {
    sf::Vector2i mousePos = sf::Mouse::getPosition(*window);
    sf::Vector2f worldPos = window->mapPixelToCoords(mousePos);

    for (size_t i = 0; i < _menu_options.size(); ++i) {
        sf::FloatRect bounds = _menu_options[i].getGlobalBounds();
        if (bounds.contains(worldPos)) {
            _selected_index = i;
            if (_selected_index == 0) {
                // Play selected
            } else if (_selected_index == 1) {
                // Options selected
            } else if (_selected_index == 2) {
                window->close();
            }
            break;
        }
    }
}

void menu::update_hover(std::unique_ptr<sf::RenderWindow>& window) {
    sf::Vector2i mouse_pos = sf::Mouse::getPosition(*window);
    sf::Vector2f world_pos = window->mapPixelToCoords(mouse_pos);

    bool hover_detected = false;

    for (size_t i = 0; i < _menu_options.size(); ++i) {
        sf::FloatRect bounds = _menu_options[i].getGlobalBounds();
        if (bounds.contains(world_pos)) {
            if (_selected_index != i) {
                _menu_options[_selected_index].setFillColor(sf::Color::White);
                _selected_index = i;
                _menu_options[_selected_index].setFillColor(sf::Color::Red);
            }
            hover_detected = true;
            break;
        }
    }

    if (!hover_detected && _selected_index >= 0 && _selected_index < _menu_options.size()) {
        _menu_options[_selected_index].setFillColor(sf::Color::Red);
    }
}
