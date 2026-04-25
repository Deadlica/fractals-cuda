#include <GUI/Widgets/dropdown.h>

static constexpr float FIELD_H = 26.0f;
static const std::string EMPTY_STR;

dropdown::dropdown(const sf::Font& font, const std::string& label,
                   const std::vector<std::string>& options, int initial_index,
                   float field_width)
    : _font(&font), _options(options), _index(initial_index),
      _open(false), _field_width(field_width), _hover(-1) {
    _label.setFont(font);
    _label.setString(label);
    _label.setCharacterSize(16);
    _label.setFillColor(sf::Color::Black);

    _field.setSize(sf::Vector2f(field_width, FIELD_H));
    _field.setFillColor(sf::Color::White);
    _field.setOutlineColor(sf::Color::Black);
    _field.setOutlineThickness(1);

    _field_text.setFont(font);
    _field_text.setCharacterSize(14);
    _field_text.setFillColor(sf::Color::Black);

    _caret.setFont(font);
    _caret.setString("v");
    _caret.setCharacterSize(14);
    _caret.setFillColor(sf::Color::Black);

    rebuild_items();
}

void dropdown::rebuild_items() {
    _item_boxes.clear();
    _item_texts.clear();
    for (const auto& opt : _options) {
        sf::RectangleShape box(sf::Vector2f(_field_width, FIELD_H));
        box.setFillColor(sf::Color::White);
        box.setOutlineColor(sf::Color::Black);
        box.setOutlineThickness(1);
        _item_boxes.push_back(box);

        sf::Text t;
        t.setFont(*_font);
        t.setString(opt);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color::Black);
        _item_texts.push_back(t);
    }
    if (_index >= 0 && static_cast<size_t>(_index) < _options.size()) {
        _field_text.setString(_options[_index]);
    } else {
        _field_text.setString("");
    }
}

void dropdown::set_options(const std::vector<std::string>& options, int initial_index) {
    _options = options;
    _index = initial_index;
    _open = false;
    rebuild_items();
    // positions will be fixed up when set_position is called
}

void dropdown::set_position(float x, float y) {
    _label.setPosition(x, y);
    float field_y = y + 20;
    _field.setPosition(x, field_y);
    _field_text.setPosition(x + 6, field_y + 4);
    _caret.setPosition(x + _field_width - 14, field_y + 4);

    for (size_t i = 0; i < _item_boxes.size(); i++) {
        float iy = field_y + FIELD_H + i * FIELD_H;
        _item_boxes[i].setPosition(x, iy);
        _item_texts[i].setPosition(x + 6, iy + 4);
    }
}

sf::FloatRect dropdown::bounds() const {
    // bounds reported here cover label + collapsed field only.
    sf::FloatRect lb = _label.getGlobalBounds();
    sf::FloatRect fb = _field.getGlobalBounds();
    float left = std::min(lb.left, fb.left);
    float top = lb.top;
    float right = std::max(lb.left + lb.width, fb.left + fb.width);
    float bottom = fb.top + fb.height;
    return sf::FloatRect(left, top, right - left, bottom - top);
}

bool dropdown::handle_event(const sf::Event& event, const sf::RenderWindow& window) {
    if (!_visible || !_enabled) return false;

    if (event.type == sf::Event::MouseMoved) {
        if (_open) {
            sf::Vector2f p = window.mapPixelToCoords({event.mouseMove.x, event.mouseMove.y});
            _hover = -1;
            for (size_t i = 0; i < _item_boxes.size(); i++) {
                if (_item_boxes[i].getGlobalBounds().contains(p)) {
                    _hover = static_cast<int>(i);
                    break;
                }
            }
        }
        return false;
    }

    if (event.type != sf::Event::MouseButtonPressed) return false;
    if (event.mouseButton.button != sf::Mouse::Left) return false;
    sf::Vector2f p = window.mapPixelToCoords({event.mouseButton.x, event.mouseButton.y});

    if (_field.getGlobalBounds().contains(p)) {
        _open = !_open;
        return true;
    }
    if (_open) {
        for (size_t i = 0; i < _item_boxes.size(); i++) {
            if (_item_boxes[i].getGlobalBounds().contains(p)) {
                _index = static_cast<int>(i);
                _field_text.setString(_options[_index]);
                _open = false;
                return true;
            }
        }
        _open = false;  // click outside closes
    }
    return false;
}

const std::string& dropdown::selected_value() const {
    if (_index < 0 || static_cast<size_t>(_index) >= _options.size()) return EMPTY_STR;
    return _options[_index];
}

void dropdown::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    if (!_visible) return;
    target.draw(_label, states);
    target.draw(_field, states);
    target.draw(_field_text, states);
    target.draw(_caret, states);
    if (_open) {
        for (size_t i = 0; i < _item_boxes.size(); i++) {
            sf::RectangleShape box = _item_boxes[i];
            if (static_cast<int>(i) == _hover) box.setFillColor(sf::Color(220, 230, 255));
            target.draw(box, states);
            target.draw(_item_texts[i], states);
        }
    }
}
