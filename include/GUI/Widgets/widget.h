#ifndef WIDGET_H
#define WIDGET_H

// SFML
#include <SFML/Graphics.hpp>

class widget : public sf::Drawable {
public:
    widget() : _visible(true), _enabled(true) {}
    virtual ~widget() = default;

    // Feed an event. Returns true if the widget consumed it.
    virtual bool handle_event(const sf::Event& event, const sf::RenderWindow& window) = 0;

    virtual void set_position(float x, float y) = 0;
    virtual sf::FloatRect bounds() const = 0;

    void set_visible(bool v) { _visible = v; }
    bool visible() const { return _visible; }
    void set_enabled(bool e) { _enabled = e; }
    bool enabled() const { return _enabled; }

protected:
    bool _visible;
    bool _enabled;
};

#endif // WIDGET_H
