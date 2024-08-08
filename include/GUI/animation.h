#ifndef ANIMATION_H
#define ANIMATION_H

// SFML
#include <SFML/Graphics.hpp>

// std
#include <vector>
#include <cmath>


class animation : public sf::Drawable, public sf::Transformable {
private:
    struct MovingCircle;
    struct RotatingSquare;
public:
    enum class Type { CIRCLES, SQUARES, WAVES };

    explicit animation(Type type);
    void initialize(const sf::RenderWindow& window);
    void update(const sf::RenderWindow& window);

private:
    void initialize_circles(const sf::RenderWindow& window);
    void initialize_squares(const sf::RenderWindow& window);
    void initialize_waves(const sf::RenderWindow& window);

    void update_circles(const sf::RenderWindow& window);
    void update_squares(const sf::RenderWindow& window);
    void update_waves(const sf::RenderWindow& window);
    void update_gradient(RotatingSquare& square);

    void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

    struct MovingCircle {
        sf::CircleShape shape;
        sf::Vector2f velocity;
    };

    struct RotatingSquare {
        sf::RectangleShape shape;
        sf::VertexArray gradient_vertices;
        float angle;
        float angular_velocity;
        sf::Vector2f velocity;
        sf::Color color1;
        sf::Color color2;
    };

    Type _type;
    sf::Clock _clock;
    sf::Clock _gradient_clock;

    const int NUM_CIRCLES = 50;
    const int NUM_SQUARES = 40;
    const int NUM_POINTS = 100;
    const double CIRCLE_RADIUS = 10.0;
    const double SQUARE_SIZE = 50.0;
    const double WAVE_HEIGHT = 50.0;
    const double WAVE_SPEED = 100.0;

    std::vector<MovingCircle> _circles;
    std::vector<RotatingSquare> _squares;
    sf::VertexArray _wave;
};

#endif // ANIMATION_H
