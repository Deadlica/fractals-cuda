#ifndef MENU_H
#define MENU_H

// Project
#include <Fractal/Params/FractalParams.h>
#include <GUI/animation.h>

// SFML
#include <SFML/Graphics.hpp>

// std
#include <vector>
#include <string>
#include <unordered_map>

class menu {
public:
    menu(float width, float height);
    ~menu();

    enum class fractal { MANDELBROT, NEWTON, BURNING_SHIP, JULIA, SIERPINSKI };

    void run(std::unique_ptr<sf::RenderWindow>& window, FractalParams& params);
    menu::fractal selected_fractal() const;

private:
    enum main_buttons { START, FRACTALS, OPTIONS, EXIT };
    enum fractal_buttons { MANDELBROT, NEWTON, BURNING_SHIP, JULIA, SIERPINSKI };
    enum option_buttons { SIZE, MAX_ITER }; // wip
    enum class mode { MAIN, FRACTALS, OPTIONS };

    sf::RenderWindow* _window;
    fractal _current_fractal;
    mode _current_mode;
    bool _is_closing;

    sf::Font _font;
    animation _background;
    sf::Clock _clock;

    std::unordered_map<mode, std::vector<sf::Text>> _menu_texts;
    std::unordered_map<mode, std::vector<sf::RectangleShape>> _menu_boxes;
    std::unordered_map<mode, int> _begin;
    std::unordered_map<mode, int> _end;
    std::unordered_map<mode, int> _selected_index;

    sf::Color _box_color;
    sf::Color _text_color;
    sf::Color _selected_text_color;
    sf::Color _selected_box_color;

    void init_menu(float width, float height);
    void draw();
    void move_up();
    void move_down();
    void handle_mouse_click();
    void action(int index);
    void load_fractals_menu();
    void load_options_menu();
    void move_to_button(int index);
    void update_hover();
};

#endif // MENU_H
