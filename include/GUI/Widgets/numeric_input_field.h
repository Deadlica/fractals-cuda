#ifndef NUMERIC_INPUT_FIELD_H
#define NUMERIC_INPUT_FIELD_H

// Project
#include <GUI/Widgets/widget.h>

class numeric_input_field : public widget {
public:
    enum class kind { INT, DOUBLE };

    numeric_input_field(const sf::Font& font, const std::string& label,
                        kind k, double initial, float field_width);

    bool handle_event(const sf::Event& event, const sf::RenderWindow& window) override;
    void set_position(float x, float y) override;
    sf::FloatRect bounds() const override;

    // Commits the current text to the underlying numeric value; returns it.
    // On parse failure, keeps the previous value and resets the text.
    int    as_int() const;
    double as_double() const;

    void set_focus(bool f);
    bool focused() const { return _focused; }

private:
    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    void refresh_text();

    sf::Text _label;
    sf::RectangleShape _field;
    sf::Text _value_text;
    kind _kind;
    std::string _buf;
    bool _focused;
};

#endif // NUMERIC_INPUT_FIELD_H
