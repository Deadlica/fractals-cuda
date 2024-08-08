// Project
#include <GUI/animation.h>

// std
#include <functional>

animation::animation(Type type): _type(type) {}

void animation::initialize(const sf::RenderWindow& window) {
    switch (_type) {
    case Type::CIRCLES:
        initialize_circles(window);
        break;
    case Type::SQUARES:
        initialize_squares(window);
        break;
    case Type::WAVES:
        initialize_waves(window);
        break;
    }
}

void animation::update(const sf::RenderWindow& window) {
    switch (_type) {
    case Type::CIRCLES:
        update_circles(window);
        break;
    case Type::SQUARES:
        std::for_each(_squares.begin(), _squares.end(), [this](RotatingSquare& square) {this->update_gradient(square);});
        update_squares(window);
        break;
    case Type::WAVES:
        update_waves(window);
        break;
    }
}

void animation::initialize_circles(const sf::RenderWindow& window) {
    _circles.clear();
    for (int i = 0; i < NUM_CIRCLES; ++i) {
        MovingCircle circle;
        circle.shape.setRadius(CIRCLE_RADIUS);
        circle.shape.setFillColor(sf::Color::White);
        circle.shape.setPosition(
        std::rand() % window.getSize().x,
        std::rand() % window.getSize().y
        );
        circle.velocity = sf::Vector2f(
        (std::rand() % 200 - 100),
        (std::rand() % 200 - 100)
        );
        _circles.push_back(circle);
    }
}

void animation::initialize_squares(const sf::RenderWindow& window) {
    _squares.clear();
    for (int i = 0; i < NUM_SQUARES; ++i) {
        RotatingSquare square;
        square.shape.setSize(sf::Vector2f(SQUARE_SIZE, SQUARE_SIZE));
        square.shape.setOrigin(SQUARE_SIZE / 2, SQUARE_SIZE / 2);
        square.shape.setPosition(
        std::rand() % window.getSize().x,
        std::rand() % window.getSize().y
        );
        square.angle = std::rand() % 360;
        square.angular_velocity = (std::rand() % 100 - 50);
        square.velocity = sf::Vector2f(
        (std::rand() % 200 - 100),
        (std::rand() % 200 - 100)
        );

        square.gradient_vertices.setPrimitiveType(sf::TriangleStrip);
        square.gradient_vertices.resize(std::rand() % 4 + 3);
        size_t vertices = square.gradient_vertices.getVertexCount();
        size_t v = 0;
        int divider = std::rand() % 6 + 1 ;
        square.gradient_vertices[v++].position = sf::Vector2f(-SQUARE_SIZE / divider, -SQUARE_SIZE / divider);
        divider = std::rand() % 6 + 1;
        square.gradient_vertices[v++].position = sf::Vector2f(SQUARE_SIZE / divider, -SQUARE_SIZE / divider);
        divider = std::rand() % 6 + 1;
        square.gradient_vertices[v++].position = sf::Vector2f(SQUARE_SIZE / divider, SQUARE_SIZE / divider);
        if (v < vertices) {
            divider = std::rand() % 6 + 1;
            square.gradient_vertices[v++].position = sf::Vector2f(-SQUARE_SIZE / divider, SQUARE_SIZE / divider);
        }
        if (v < vertices) {
            divider = std::rand() % 6 + 1;
            square.gradient_vertices[v++].position = sf::Vector2f(-SQUARE_SIZE / divider, -SQUARE_SIZE / divider);
        }
        if (v < vertices) {
            divider = std::rand() % 6 + 1;
            square.gradient_vertices[v++].position = sf::Vector2f(SQUARE_SIZE / divider, -SQUARE_SIZE / divider);
        }

        update_gradient(square);

        _squares.push_back(square);
    }
}

void animation::initialize_waves(const sf::RenderWindow& window) {
    _wave.clear();
    _wave.setPrimitiveType(sf::LinesStrip);
    _wave.resize(NUM_POINTS);

    for (int i = 0; i < NUM_POINTS; ++i) {
        _wave[i].position = sf::Vector2f(i * (window.getSize().x / (NUM_POINTS - 1)), window.getSize().y / 2);
        _wave[i].color = sf::Color::White;
    }
}

void animation::update_circles(const sf::RenderWindow& window) {
    float delta_time = _clock.restart().asSeconds();

    for (auto& circle : _circles) {
        sf::Vector2f pos = circle.shape.getPosition();
        pos += circle.velocity * delta_time;

        if (pos.x < 0) pos.x = window.getSize().x;
        if (pos.x > window.getSize().x) pos.x = 0;
        if (pos.y < 0) pos.y = window.getSize().y;
        if (pos.y > window.getSize().y) pos.y = 0;

        circle.shape.setPosition(pos);
    }
}

void animation::update_squares(const sf::RenderWindow& window) {
    float delta_time = _clock.restart().asSeconds();

    for (auto& square : _squares) {
        // Update angle
        square.angle += square.angular_velocity * delta_time;
        square.shape.setRotation(square.angle);

        // Update position
        sf::Vector2f position = square.shape.getPosition();
        position += square.velocity * delta_time;

        // Wrap around the screen
        if (position.x < 0) position.x = static_cast<float>(window.getSize().x);
        if (position.x > window.getSize().x) position.x = 0;
        if (position.y < 0) position.y = static_cast<float>(window.getSize().y);
        if (position.y > window.getSize().y) position.y = 0;

        square.shape.setPosition(position);
    }
}

void animation::update_waves(const sf::RenderWindow& window) {
    double time = _clock.getElapsedTime().asSeconds();

    for (int i = 0; i < NUM_POINTS; ++i) {
        float x = _wave[i].position.x;
        float y = window.getSize().y / 2 + WAVE_HEIGHT * std::sin((x + WAVE_SPEED * time) * 0.01);
        _wave[i].position = sf::Vector2f(x, y);
    }
}

void animation::update_gradient(RotatingSquare& square) {
    float time = _gradient_clock.getElapsedTime().asSeconds();

    size_t vertices = square.gradient_vertices.getVertexCount();
    size_t v = 0;
    square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * sin(time));
    square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * cos(time));
    square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * sin(time + 3.14f));
    square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * cos(time));
    square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * sin(time));
    square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * cos(time + 3.14f));
    square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * sin(time + 1.57f));
    square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f));
    square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * sin(time + 1.57f + 3.14f));
    if (v < vertices) {
        square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f));
        square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * sin(time + 1.57f));
        square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f + 3.14f));
    }
    if (v < vertices) {
        square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f));
        square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * sin(time + 1.57f));
        square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f + 3.14f));
    }
    if (v < vertices) {
        square.gradient_vertices[v].color.r = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f));
        square.gradient_vertices[v].color.g = static_cast<sf::Uint8>(128 + 127 * sin(time + 1.57f));
        square.gradient_vertices[v++].color.b = static_cast<sf::Uint8>(128 + 127 * cos(time + 1.57f + 3.14f));
    }
}

void animation::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.clear(sf::Color(40, 44, 52));
    switch (_type) {
    case Type::CIRCLES:
        for (const MovingCircle& circle : _circles) {
            target.draw(circle.shape, states);
        }
        break;
    case Type::SQUARES:
        for (const RotatingSquare& square : _squares) {
            states.transform = square.shape.getTransform();
            target.draw(square.gradient_vertices, states);
        }
        break;
    case Type::WAVES:
        target.draw(_wave, states);
        break;
    }

}
