// Project
#include <GUI/coordinate_label.h>

coordinate_label::coordinate_label(): coordinate_label(8) {}

coordinate_label::coordinate_label(unsigned long precision) {
    _font.loadFromFile("fonts/arial.ttf");
    _label.setFont(_font);
    _label.setCharacterSize(14);
    _label.setFillColor(sf::Color::Black);

    _oss << std::setprecision(precision);
    {
        std::ostringstream temp_oss;
        temp_oss << std::fixed << std::setprecision(precision + 6) << 0.0;
        _label.setString("x = " + temp_oss.str() + "\ny = " + temp_oss.str());
    }

    _border = sf::RectangleShape(sf::Vector2f(
            _label.getGlobalBounds().width + 10, _label.getGlobalBounds().height + 10)
    );
    _border.setFillColor(sf::Color::White);
    _border.setOutlineThickness(1);
    _border.setOutlineColor(sf::Color::Black);
}

void coordinate_label::set_position(float x, float y) {
    _border.setPosition(x, y - _border.getSize().y);
    _label.setPosition(x + 5, y - _border.getSize().y + 2);
}

void coordinate_label::set_coordinate_string(double x, double y) {
    _oss.str("");
    _oss << "x = " << x << "\ny = " << y;
    _label.setString(_oss.str());
}

void coordinate_label::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(_border, states);
    target.draw(_label, states);
}
