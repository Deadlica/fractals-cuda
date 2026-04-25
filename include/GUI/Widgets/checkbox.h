#ifndef CHECKBOX_H
#define CHECKBOX_H

// Project
#include <GUI/Widgets/widget.h>

class checkbox : public widget {
public:
    checkbox(const sf::Font& font, const std::string& label, bool initial);

    bool handle_event(const sf::Event& event, const sf::RenderWindow& window) override;
    void set_position(float x, float y) override;
    sf::FloatRect bounds() const override;

    bool checked() const { return _checked; }
    void set_checked(bool v) { _checked = v; }

private:
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

    sf::RectangleShape _box;
    sf::RectangleShape _fill;
    sf::Text _label;
    bool _checked;
};

#endif // CHECKBOX_H
