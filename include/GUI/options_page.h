#ifndef OPTIONS_PAGE_H
#define OPTIONS_PAGE_H

// Project
#include <Fractal/fractal_type.h>
#include <GUI/Widgets/checkbox.h>
#include <GUI/Widgets/dropdown.h>
#include <GUI/Widgets/numeric_input_field.h>

// SFML
#include <SFML/Graphics.hpp>

// std
#include <functional>
#include <memory>

// A bundle of user-configurable settings passed into the options page.
struct app_settings {
    int width;
    int height;
    std::string pattern;   // empty = none
    std::string theme;     // empty = default
    int max_iter;
    double zoom_factor;
    bool smooth;
    double julia_c_re;
    double julia_c_im;
};

inline bool operator==(const app_settings& a, const app_settings& b) {
    return a.width == b.width && a.height == b.height &&
           a.pattern == b.pattern && a.theme == b.theme &&
           a.max_iter == b.max_iter && a.zoom_factor == b.zoom_factor &&
           a.smooth == b.smooth && a.julia_c_re == b.julia_c_re &&
           a.julia_c_im == b.julia_c_im;
}
inline bool operator!=(const app_settings& a, const app_settings& b) { return !(a == b); }

// Runs a modal options UI inside the given window.
// When the user clicks Apply, `on_apply(new_settings)` is invoked immediately so
// the change takes effect while the menu is still on screen (allowing a window
// resize to rescale everything). Returns the final settings the user left with.
void run_options_page(sf::RenderWindow& window, const app_settings& initial,
                      fractal_type current_fractal,
                      const std::function<void(const app_settings&)>& on_apply);

#endif // OPTIONS_PAGE_H
