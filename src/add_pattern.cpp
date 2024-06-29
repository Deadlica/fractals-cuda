#include "add_pattern.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

add_pattern::add_pattern(double x_min, double x_max, double y_min, double y_max, int px, int py, const std::string& filename):
_width(300), _height(140),
_window(sf::VideoMode(_width, _height), "Save Pattern", sf::Style::Titlebar | sf::Style::Close),
_x_min(x_min), _x_max(x_max), _y_min(y_min), _y_max(y_max), _file_name(filename) {

    _window.setPosition(sf::Vector2i(px - _width / 2, py - _height / 2));
}

void add_pattern::run() {
    sf::Font font;
    if (!font.loadFromFile("fonts/arial.ttf")) {
        return;
    }

    sf::Text label_text;
    label_text.setFont(font);
    label_text.setString("Enter pattern name:");
    label_text.setCharacterSize(20);
    label_text.setFillColor(sf::Color::Black);
    label_text.setPosition(10, 10);

    sf::RectangleShape input_border(sf::Vector2f(_width - 20, 30));
    input_border.setPosition(10, 40);
    input_border.setFillColor(sf::Color::White);
    input_border.setOutlineThickness(1);
    input_border.setOutlineColor(sf::Color::Black);

    sf::Text input_text;
    input_text.setFont(font);
    input_text.setCharacterSize(16);
    input_text.setFillColor(sf::Color::Black);
    input_text.setPosition(15, 45);

    sf::RectangleShape button(sf::Vector2f(80, 30));
    button.setPosition(_width / 2 - 40, 90);
    button.setFillColor(sf::Color::White);
    button.setOutlineThickness(1);
    button.setOutlineColor(sf::Color::Black);

    sf::Text button_text;
    button_text.setFont(font);
    button_text.setString("Ok");
    button_text.setCharacterSize(16);
    button_text.setFillColor(sf::Color::Black);
    button_text.setPosition(_width / 2 - 10, 95);

    bool button_clicked = false;
    sf::Clock animation_clock;

    sf::Cursor arrow_cursor;
    sf::Cursor hand_cursor;
    sf::Cursor text_cursor;

    if (!arrow_cursor.loadFromSystem(sf::Cursor::Arrow) ||
        !hand_cursor.loadFromSystem(sf::Cursor::Hand)   ||
        !text_cursor.loadFromSystem(sf::Cursor::Text)) {
        std::cout << "Could not load cursors" << std::endl;
        return;
    }

    while(_window.isOpen()) {
        sf::Event event;
        while (_window.pollEvent(event)) {
            unsigned int c;
            switch (event.type) {
            case sf::Event::Closed:
                _window.close();
                break;
            case sf::Event::TextEntered:
                c = event.text.unicode;
                if (!std::isalnum(c) && c != '-' && c != '_' && c != '\b') break;
                if (event.text.unicode == '\b' && !_input_text.empty()) { // Backspace
                    _input_text.pop_back();
                }
                else if (input_text.getGlobalBounds().width < input_border.getSize().x - 20) {
                    _input_text += static_cast<char>(event.text.unicode);
                }
                input_text.setString(_input_text);
                break;
            case sf::Event::KeyPressed:
                switch (event.key.code) {
                case sf::Keyboard::Enter:
                    save_coords();
                    button_clicked = true;
                    animation_clock.restart();
                    break;
                case sf::Keyboard::Escape:
                    _window.close();
                    break;
                default:
                    break;
                }
                break;
            case sf::Event::MouseButtonPressed:
                if (event.mouseButton.button == sf::Mouse::Left) {
                    sf::Vector2i mouse_pos = sf::Mouse::getPosition(_window);
                    if (button.getGlobalBounds().contains(mouse_pos.x, mouse_pos.y)) {
                        save_coords();
                        button_clicked = true;
                        animation_clock.restart();
                    }
                }
                break;
            case sf::Event::MouseMoved:
            {
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(_window);
                if  (input_border.getGlobalBounds().contains(mouse_pos.x, mouse_pos.y)) {
                    _window.setMouseCursor(text_cursor);
                }
                else if (button.getGlobalBounds().contains(mouse_pos.x, mouse_pos.y)) {
                    button.setFillColor(sf::Color(200, 200, 200));
                    _window.setMouseCursor(hand_cursor);
                }
                else {
                    button.setFillColor(sf::Color::White);
                    _window.setMouseCursor(arrow_cursor);
                }
                break;
            }
            case sf::Event::Resized:
                _window.setSize(sf::Vector2<unsigned int>(_width, _height));
                break;
            default:
                break;
            }
        }

        if (button_clicked) {
            sf::Time elapsed = animation_clock.getElapsedTime();
            if (elapsed.asMilliseconds() < 100) {
                button.setFillColor(sf::Color::Green);
            }
            else {
                button.setFillColor(sf::Color::White);
                _window.close();
            }
        }

        _window.clear(sf::Color::White);
        _window.draw(label_text);
        _window.draw(input_border);
        _window.draw(input_text);
        _window.draw(button);
        _window.draw(button_text);
        _window.display();
    }
}
void add_pattern::save_coords() {
    std::vector<std::string> lines;
    std::ostringstream oss;
    oss << std::setprecision(16) << _input_text + " " << _x_min << " " << _y_min << " " << _x_max << " " << _y_max;
    std::string new_pattern = oss.str();
    std::ifstream ifile(_file_name);
    if (ifile.is_open()) {
        std::string line;
        while (!ifile.eof()) {
            std::getline(ifile, line);
            if (line.empty()) continue;
            if (line.size() >= _input_text.size()) {
                lines.push_back(
                        line.substr(0, _input_text.size()) == _input_text ? new_pattern : line
                );
            }
            else {
                lines.push_back(line);
            }
        }
        ifile.close();
    }
    if (std::find(lines.begin(), lines.end(), new_pattern) == lines.end()) {
        lines.push_back(new_pattern);
    }

    std::ofstream ofile(_file_name);
    if (!ofile.is_open()) {
        std::cout << "Failed to save pattern " << _input_text << std::endl;
        return;
    }

    for (const std::string& line : lines) {
        ofile << line << std::endl;
    }

    ofile.close();
}