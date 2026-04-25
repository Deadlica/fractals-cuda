#include <GUI/Widgets/checkbox.h>

static constexpr float BOX_SIZE = 20.0f;

checkbox::checkbox(const sf::Font& font, const std::string& label, bool initial)
    : _checked(initial) {
    _box.setSize(sf::Vector2f(BOX_SIZE, BOX_SIZE));
    _box.setFillColor(sf::Color::White);
    _box.setOutlineColor(sf::Color::Black);
    _box.setOutlineThickness(1);

    _fill.setSize(sf::Vector2f(BOX_SIZE - 6, BOX_SIZE - 6));
    _fill.setFillColor(sf::Color(40, 120, 200));

    _label.setFont(font);
    _label.setString(label);
    _label.setCharacterSize(16);
    _label.setFillColor(sf::Color::Black);
}

void checkbox::set_position(float x, float y) {
    _box.setPosition(x, y);
    _fill.setPosition(x + 3, y + 3);
    _label.setPosition(x + BOX_SIZE + 8, y + 1);
}

sf::FloatRect checkbox::bounds() const {
    sf::FloatRect b = _box.getGlobalBounds();
    b.width += 8 + _label.getGlobalBounds().width;
    return b;
}

bool checkbox::handle_event(const sf::Event& event, const sf::RenderWindow& window) {
    if (!_visible || !_enabled) return false;
    if (event.type != sf::Event::MouseButtonPressed) return false;
    if (event.mouseButton.button != sf::Mouse::Left) return false;
    sf::Vector2f p = window.mapPixelToCoords({event.mouseButton.x, event.mouseButton.y});
    if (!bounds().contains(p)) return false;
    _checked = !_checked;
    return true;
}

void checkbox::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    if (!_visible) return;
    target.draw(_box, states);
    if (_checked) target.draw(_fill, states);
    target.draw(_label, states);
}
