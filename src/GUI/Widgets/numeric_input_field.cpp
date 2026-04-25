#include <GUI/Widgets/numeric_input_field.h>
#include <sstream>
#include <iomanip>
#include <cstdlib>

static constexpr float FIELD_H = 26.0f;

numeric_input_field::numeric_input_field(const sf::Font& font, const std::string& label,
                                         kind k, double initial, float field_width)
    : _kind(k), _focused(false) {
    _label.setFont(font);
    _label.setString(label);
    _label.setCharacterSize(16);
    _label.setFillColor(sf::Color::Black);

    _field.setSize(sf::Vector2f(field_width, FIELD_H));
    _field.setFillColor(sf::Color::White);
    _field.setOutlineColor(sf::Color::Black);
    _field.setOutlineThickness(1);

    _value_text.setFont(font);
    _value_text.setCharacterSize(14);
    _value_text.setFillColor(sf::Color::Black);

    std::ostringstream oss;
    if (k == kind::INT) oss << static_cast<int>(initial);
    else                oss << std::setprecision(10) << initial;
    _buf = oss.str();
    refresh_text();
}

void numeric_input_field::refresh_text() {
    _value_text.setString(_buf + (_focused ? "|" : ""));
}

void numeric_input_field::set_position(float x, float y) {
    _label.setPosition(x, y);
    _field.setPosition(x, y + 20);
    _value_text.setPosition(x + 6, y + 24);
}

sf::FloatRect numeric_input_field::bounds() const {
    sf::FloatRect lb = _label.getGlobalBounds();
    sf::FloatRect fb = _field.getGlobalBounds();
    float left = std::min(lb.left, fb.left);
    float top = lb.top;
    float right = std::max(lb.left + lb.width, fb.left + fb.width);
    float bottom = fb.top + fb.height;
    return sf::FloatRect(left, top, right - left, bottom - top);
}

void numeric_input_field::set_focus(bool f) {
    _focused = f;
    refresh_text();
}

bool numeric_input_field::handle_event(const sf::Event& event, const sf::RenderWindow& window) {
    if (!_visible || !_enabled) return false;

    if (event.type == sf::Event::MouseButtonPressed &&
        event.mouseButton.button == sf::Mouse::Left) {
        sf::Vector2f p = window.mapPixelToCoords({event.mouseButton.x, event.mouseButton.y});
        bool hit = _field.getGlobalBounds().contains(p);
        set_focus(hit);
        return hit;
    }

    if (!_focused) return false;

    if (event.type == sf::Event::TextEntered) {
        unsigned int c = event.text.unicode;
        if (c == '\b') {                          // backspace
            if (!_buf.empty()) _buf.pop_back();
        } else if (c == '\r' || c == '\n') {      // enter
            set_focus(false);
            return true;
        } else if ((c >= '0' && c <= '9')) {
            _buf += static_cast<char>(c);
        } else if (c == '-' && _buf.empty()) {
            _buf += '-';
        } else if (c == '.' && _kind == kind::DOUBLE &&
                   _buf.find('.') == std::string::npos) {
            _buf += '.';
        } else {
            return false;
        }
        refresh_text();
        return true;
    }

    return false;
}

int numeric_input_field::as_int() const {
    try { return std::stoi(_buf); } catch (...) { return 0; }
}

double numeric_input_field::as_double() const {
    try { return std::stod(_buf); } catch (...) { return 0.0; }
}

void numeric_input_field::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    if (!_visible) return;
    target.draw(_label, states);
    sf::RectangleShape field = _field;
    if (_focused) field.setOutlineColor(sf::Color(40, 120, 200));
    target.draw(field, states);
    target.draw(_value_text, states);
}
