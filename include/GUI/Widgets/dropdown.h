#ifndef DROPDOWN_H
#define DROPDOWN_H

// Project
#include <GUI/Widgets/widget.h>

// std
#include <vector>

class dropdown : public widget {
public:
    dropdown(const sf::Font& font, const std::string& label,
             const std::vector<std::string>& options, int initial_index,
             float field_width);

    bool handle_event(const sf::Event& event, const sf::RenderWindow& window) override;
    void set_position(float x, float y) override;
    sf::FloatRect bounds() const override;

    int selected_index() const { return _index; }
    const std::string& selected_value() const;
    void set_options(const std::vector<std::string>& options, int initial_index);
    bool is_open() const { return _open; }

private:
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void rebuild_items();

    const sf::Font* _font;
    sf::Text _label;
    sf::RectangleShape _field;
    sf::Text _field_text;
    sf::Text _caret;
    std::vector<std::string> _options;
    std::vector<sf::RectangleShape> _item_boxes;
    std::vector<sf::Text> _item_texts;
    int _index;
    bool _open;
    float _field_width;
    int _hover;
};

#endif // DROPDOWN_H
