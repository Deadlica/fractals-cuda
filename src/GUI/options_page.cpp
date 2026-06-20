// Project
#include <GUI/options_page.h>
#include <CLI/cli.h>
#include <Util/globals.h>
#include <Util/util.h>

// std
#include <algorithm>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace {

const std::vector<std::pair<int,int>> SIZE_OPTIONS = {
    {600, 400}, {800, 600}, {1200, 800}, {1600, 1000},
    {1920, 1080}, {2560, 1440}, {3840, 2160}
};

std::string size_label(int w, int h) {
    return std::to_string(w) + " x " + std::to_string(h);
}

int find_size_index(int w, int h) {
    for (size_t i = 0; i < SIZE_OPTIONS.size(); i++)
        if (SIZE_OPTIONS[i].first == w && SIZE_OPTIONS[i].second == h) return static_cast<int>(i);
    return 2;
}

std::vector<std::string> list_themes() {
    std::vector<std::string> out = {"(default)"};
    DIR* d = opendir(THEMES_PATH.c_str());
    if (!d) return out;
    struct dirent* ent;
    while ((ent = readdir(d))) {
        std::string name = ent->d_name;
        if (util::ends_with(name, ".mbt")) {
            out.push_back(name.substr(0, name.size() - 4));
        }
    }
    closedir(d);
    std::sort(out.begin() + 1, out.end());
    return out;
}

std::vector<std::string> list_patterns(fractal_type ft) {
    std::vector<std::string> out = {"(none)"};
    for (const auto& kv : goals) {
        if (kv.second.type == ft) out.push_back(kv.first);
    }
    std::sort(out.begin() + 1, out.end());
    return out;
}

int index_of(const std::vector<std::string>& v, const std::string& s) {
    for (size_t i = 0; i < v.size(); i++) if (v[i] == s) return static_cast<int>(i);
    return 0;
}

struct text_button {
    sf::RectangleShape box;
    sf::Text text;

    text_button(const sf::Font& font, const std::string& label, float w, float h) {
        box.setSize(sf::Vector2f(w, h));
        box.setFillColor(sf::Color::White);
        box.setOutlineColor(sf::Color::Black);
        box.setOutlineThickness(1);
        text.setFont(font);
        text.setString(label);
        text.setCharacterSize(16);
        text.setFillColor(sf::Color::Black);
    }
    void set_position(float x, float y) {
        box.setPosition(x, y);
        sf::FloatRect tb = text.getLocalBounds();
        text.setPosition(x + (box.getSize().x - tb.width) / 2 - tb.left,
                         y + (box.getSize().y - tb.height) / 2 - tb.top);
    }
    bool contains(const sf::RenderWindow& w, const sf::Event::MouseButtonEvent& m) const {
        sf::Vector2f p = w.mapPixelToCoords({m.x, m.y});
        return box.getGlobalBounds().contains(p);
    }
    void draw(sf::RenderTarget& t) const { t.draw(box); t.draw(text); }
    void set_hover(bool h) { box.setFillColor(h ? sf::Color(220, 230, 255) : sf::Color::White); }
};

int confirm_apply_popup(sf::RenderWindow& parent, const sf::Font& font) {
    const int pw = 360, ph = 140;
    sf::Vector2i ppos = parent.getPosition();
    sf::Vector2u psize = parent.getSize();

    sf::RenderWindow dlg(sf::VideoMode(pw, ph), "Apply changes?",
                         sf::Style::Titlebar);
    dlg.setPosition(sf::Vector2i(ppos.x + (int)psize.x / 2 - pw / 2,
                                 ppos.y + (int)psize.y / 2 - ph / 2));

    sf::Text msg;
    msg.setFont(font);
    msg.setString("Apply changes before leaving?");
    msg.setCharacterSize(18);
    msg.setFillColor(sf::Color::Black);
    sf::FloatRect mb = msg.getLocalBounds();
    msg.setPosition((pw - mb.width) / 2 - mb.left, 24);

    text_button cancel(font, "Cancel", 110, 32);
    text_button apply (font, "Apply",  110, 32);
    cancel.set_position(pw / 2 - 120, ph - 50);
    apply .set_position(pw / 2 + 10,  ph - 50);

    while (dlg.isOpen()) {
        sf::Event event;
        while (dlg.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                dlg.close();
                return -1;
            }
            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                if (apply.contains(dlg, event.mouseButton))  { dlg.close(); return +1; }
                if (cancel.contains(dlg, event.mouseButton)) { dlg.close(); return -1; }
            }
            if (event.type == sf::Event::MouseMoved) {
                sf::Event::MouseButtonEvent m{sf::Mouse::Left, event.mouseMove.x, event.mouseMove.y};
                cancel.set_hover(cancel.contains(dlg, m));
                apply .set_hover(apply .contains(dlg, m));
            }
        }

        dlg.clear(sf::Color::White);
        dlg.draw(msg);
        cancel.draw(dlg);
        apply.draw(dlg);
        dlg.display();
    }
    return -1;
}

} // namespace

void run_options_page(sf::RenderWindow& window, const app_settings& initial,
                      fractal_type current_fractal,
                      const std::function<void(const app_settings&)>& on_apply) {
    sf::Font font;
    if (!font.loadFromFile(FONT_PATH)) return;

    std::vector<std::string> size_opts;
    for (auto& s : SIZE_OPTIONS) size_opts.push_back(size_label(s.first, s.second));
    std::vector<std::string> theme_opts = list_themes();
    std::vector<std::string> pattern_opts = list_patterns(current_fractal);

    dropdown size_dd(font, "Size", size_opts,
                     find_size_index(initial.width, initial.height), 260);
    dropdown theme_dd(font, "Color theme", theme_opts,
                      initial.theme.empty() ? 0 : index_of(theme_opts, initial.theme), 260);
    dropdown pattern_dd(font, "Pattern", pattern_opts,
                        initial.pattern.empty() ? 0 : index_of(pattern_opts, initial.pattern), 260);
    numeric_input_field max_iter_f(font, "Max iterations", numeric_input_field::kind::INT,
                                   initial.max_iter, 120);
    numeric_input_field zoom_f(font, "Zoom factor", numeric_input_field::kind::DOUBLE,
                               initial.zoom_factor, 120);
    checkbox smooth_cb(font, "Smooth colouring", initial.smooth);
    numeric_input_field jre_f(font, "Julia c (re)", numeric_input_field::kind::DOUBLE,
                              initial.julia_c_re, 120);
    numeric_input_field jim_f(font, "Julia c (im)", numeric_input_field::kind::DOUBLE,
                              initial.julia_c_im, 120);
    numeric_input_field mn_f(font, "Multibrot power (2-8)", numeric_input_field::kind::INT,
                             initial.multibrot_n, 120);
    checkbox fullscreen_cb(font, "Fullscreen", initial.fullscreen);

    bool is_escape_time = (current_fractal == fractal_type::MANDELBROT ||
                           current_fractal == fractal_type::JULIA      ||
                           current_fractal == fractal_type::BURNING_SHIP ||
                           current_fractal == fractal_type::NEWTON      ||
                           current_fractal == fractal_type::MULTIBROT   ||
                           current_fractal == fractal_type::NOVA);
    max_iter_f.set_visible(is_escape_time);
    smooth_cb .set_visible(current_fractal == fractal_type::MANDELBROT ||
                           current_fractal == fractal_type::JULIA      ||
                           current_fractal == fractal_type::BURNING_SHIP ||
                           current_fractal == fractal_type::MULTIBROT   ||
                           current_fractal == fractal_type::NOVA);
    jre_f.set_visible(current_fractal == fractal_type::JULIA);
    jim_f.set_visible(current_fractal == fractal_type::JULIA);
    mn_f.set_visible(current_fractal == fractal_type::MULTIBROT);

    text_button apply_btn (font, "Apply",  100, 32);
    text_button cancel_btn(font, "Cancel", 100, 32);

    sf::RectangleShape bg;
    bg.setFillColor(sf::Color(235, 235, 240));

    auto layout = [&](unsigned int ww, unsigned int wh) {
        bg.setSize(sf::Vector2f(ww, wh));
        const float col_x = 40;
        float y = 60;
        const float dy = 60;
        size_dd.set_position(col_x, y);
        fullscreen_cb.set_position(col_x + 280, y + 28);  // right of size dropdown
        y += dy;
        theme_dd.set_position(col_x, y); y += dy;
        pattern_dd.set_position(col_x, y); y += dy;
        max_iter_f.set_position(col_x, y);
        zoom_f.set_position(col_x + 160, y); y += dy;
        smooth_cb.set_position(col_x, y); y += 40;
        jre_f.set_position(col_x, y);
        jim_f.set_position(col_x + 160, y); y += dy;
        mn_f.set_position(col_x, y); y += dy;

        cancel_btn.set_position(ww - 230, wh - 60);
        apply_btn .set_position(ww - 120, wh - 60);
    };
    layout(window.getSize().x, window.getSize().y);

    std::vector<widget*> base_widgets = {&smooth_cb, &max_iter_f, &zoom_f, &jre_f, &jim_f, &mn_f, &fullscreen_cb};
    std::vector<dropdown*> dd_widgets = {&size_dd, &theme_dd, &pattern_dd};

    auto current_settings = [&]() {
        app_settings s = initial;
        s.width  = SIZE_OPTIONS[size_dd.selected_index()].first;
        s.height = SIZE_OPTIONS[size_dd.selected_index()].second;
        s.theme   = theme_dd.selected_index() == 0 ? "" : theme_dd.selected_value();
        s.pattern = pattern_dd.selected_index() == 0 ? "" : pattern_dd.selected_value();
        if (is_escape_time) { s.max_iter = max_iter_f.as_int(); s.smooth = smooth_cb.checked(); }
        s.zoom_factor = zoom_f.as_double();
        if (current_fractal == fractal_type::JULIA) {
            s.julia_c_re = jre_f.as_double(); s.julia_c_im = jim_f.as_double();
        }
        if (current_fractal == fractal_type::MULTIBROT) {
            int n = mn_f.as_int();
            if (n < 2) n = 2;
            if (n > 8) n = 8;
            s.multibrot_n = n;
        }
        s.fullscreen = fullscreen_cb.checked();
        return s;
    };

    app_settings baseline = initial;

    window.setView(sf::View(sf::FloatRect(0, 0, window.getSize().x, window.getSize().y)));

    auto attempt_exit = [&](bool& should_return) {
        app_settings cur = current_settings();
        if (cur == baseline) { should_return = true; return; }
        int r = confirm_apply_popup(window, font);
        if (r > 0) { on_apply(cur); baseline = cur; }
        should_return = true;
    };

    bool done = false;
    while (!done && window.isOpen()) {
        size_dd.set_enabled(!fullscreen_cb.checked());
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)       { attempt_exit(done); break; }
            if (event.type == sf::Event::Resized) {
                window.setView(sf::View(sf::FloatRect(0, 0, event.size.width, event.size.height)));
                layout(event.size.width, event.size.height);
                continue;
            }
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) { attempt_exit(done); break; }
            if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Right) { attempt_exit(done); break; }
                if (event.mouseButton.button == sf::Mouse::Left) {
                    if (apply_btn.contains(window, event.mouseButton)) {
                        app_settings cur = current_settings();
                        if (cur != baseline) { on_apply(cur); baseline = cur; }
                        done = true; break;
                    }
                    if (cancel_btn.contains(window, event.mouseButton)) {
                        done = true; break;
                    }
                }
            }

            bool consumed = false;
            for (auto* dd : dd_widgets) if (dd->handle_event(event, window)) { consumed = true; break; }
            if (consumed) continue;
            for (auto* w : base_widgets) if (w->handle_event(event, window)) break;

            if (event.type == sf::Event::MouseMoved) {
                sf::Event::MouseButtonEvent m{sf::Mouse::Left, event.mouseMove.x, event.mouseMove.y};
                apply_btn .set_hover(apply_btn .contains(window, m));
                cancel_btn.set_hover(cancel_btn.contains(window, m));
            }
        }
        if (done) break;

        window.clear();
        window.draw(bg);
        for (auto* w : base_widgets) window.draw(*w);
        dropdown* open_dd = nullptr;
        for (auto* dd : dd_widgets) if (dd->is_open()) { open_dd = dd; break; }
        for (auto* dd : dd_widgets) if (dd != open_dd) window.draw(*dd);
        if (open_dd) window.draw(*open_dd);

        apply_btn.draw(window);
        cancel_btn.draw(window);
        window.display();
    }
}
