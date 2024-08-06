#ifndef COORDINATE_LABEL_H
#define COORDINATE_LABEL_H

#include <sstream>
#include <iomanip>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/Transformable.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderStates.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/Text.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

class coordinate_label : public sf::Drawable, public sf::Transformable {
public:
    coordinate_label();
    explicit coordinate_label(unsigned long precision);

    void set_position(float x, float y);
    void set_coordinate_string(double x, double y);

private:
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

    sf::Font _font;
    sf::Text _label;
    sf::RectangleShape _border;
    std::ostringstream _oss;
};


#endif // COORDINATE_LABEL_H
