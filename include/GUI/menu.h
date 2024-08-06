#ifndef MENU_H
#define MENU_H

#include <SFML/Graphics.hpp>
#include <vector>
#include <string>

class menu {
public:
    menu(float width, float height);
    ~menu();

    void run(sf::RenderWindow& window);

private:
    int _selected_index;
    sf::Font _font;
    std::vector<sf::Text> _menu_options;

    void init_menu(float width, float height);
    void draw(sf::RenderWindow &window);
    void move_up();
    void move_down();
    void handle_mouse_click(sf::RenderWindow &window);
    void update_hover(sf::RenderWindow &window);
};

#endif // MENU_H
