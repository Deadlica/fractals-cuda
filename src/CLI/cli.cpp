// Project
#include <CLI/cli.h>
#include <Util/util.h>

// std
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// All predefined presets are Mandelbrot views.
static goal mandelbrot_goal(double xmin, double ymin, double xmax, double ymax) {
    return goal{{xmin, ymin}, {xmax, ymax}, fractal_type::MANDELBROT, 0.0, 0.0};
}

std::unordered_map<std::string, goal> goals = {
    {"flower",    mandelbrot_goal(-1.999985885,  2e-9,        -1.999985879, -3e-9)},
    {"julia_island", mandelbrot_goal(-1.768779320, -0.001738521, -1.768778317, -0.001739281)},
    {"seahorse",  mandelbrot_goal(-0.750555615, -0.121803013, -0.736462413, -0.132368505)},
    {"starfish",  mandelbrot_goal(-0.375652034,  0.661031194, -0.372352113,  0.658557285)},
    {"sun",       mandelbrot_goal(-0.776606539, -0.136630553, -0.776579121, -0.136651108)},
    {"tendrils",  mandelbrot_goal(-0.226267721,  1.116175247, -0.226265572,  1.116173636)},
    {"tree",      mandelbrot_goal(-1.940158339, -4.9e-8,      -1.940156342, -0.000001548)}
};

std::unordered_map<std::string, goal> custom_goals;

// Pattern file line formats:
//   <name> <x_min> <y_min> <x_max> <y_max>                          (legacy, implies mandelbrot)
//   <name> <fractal_id> <x_min> <y_min> <x_max> <y_max>             (non-julia)
//   <name> julia <x_min> <y_min> <x_max> <y_max> <c_re> <c_im>      (julia)
void init_custom_cli_patterns(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string name, second;
        if (!(iss >> name >> second)) continue;

        fractal_type type;
        double x_min, y_min, x_max, y_max;
        double c_re = 0.0, c_im = 0.0;

        if (fractal_type_from_string(second, type)) {
            if (!(iss >> x_min >> y_min >> x_max >> y_max)) continue;
            if (type == fractal_type::JULIA && !(iss >> c_re >> c_im)) continue;
        } else {
            // legacy: `second` was actually x_min
            try { x_min = std::stod(second); } catch (...) { continue; }
            if (!(iss >> y_min >> x_max >> y_max)) continue;
            type = fractal_type::MANDELBROT;
        }

        goal g{{x_min, y_min}, {x_max, y_max}, type, c_re, c_im};
        custom_goals[name] = g;
        goals[name] = g;
    }
}

void cli_help() {
    std::string help_text1 = R"(
Usage: fractals [options]

Options:
    --size <dimension>      Set the width and height of the window with "x" as a separator (default: 1200x800)
    --pattern <name>        Choose a predefined pattern to zoom into (default: Mandelbrot)
                            Available patterns: flower, julia_island, seahorse, starfish, sun, tendrils, tree
)";
    std::string help_text2 = R"(
    --max-iter <value>      Set the maximum number of iterations per pixel (default: 500)
    --zoom-factor <value>   Set the zoom factor per iteration (default: 0.95)
    --smooth                Enables smoothing, reduces color bands
    --theme <name>          Use a custom defined color theme (under assets/themes/*.mbt)
    --julia-c <re,im>       Julia constant; e.g. -0.8,0.156 (default: -0.8,0.156)
    --help                  Display this help message

Hotkeys:
    S       Save pattern coordinates
    Home    Reset position
    Esc     Exit program / Open menu

Examples:
    fractals --pattern seahorse --smooth
    fractals --size 1000x1000
    fractals --pattern flower --zoom-factor 0.97
    fractals --julia-c -0.70176,-0.3842
)";

    std::string extra_patterns;
    if (!custom_goals.empty()) {
        std::unordered_map<fractal_type, std::vector<std::string>> by_type;
        for (const auto& kv : custom_goals) by_type[kv.second.type].push_back(kv.first);

        const fractal_type order[] = {
            fractal_type::MANDELBROT, fractal_type::JULIA,
            fractal_type::BURNING_SHIP, fractal_type::NEWTON,
            fractal_type::SIERPINSKI,
        };
        for (fractal_type t : order) {
            auto it = by_type.find(t);
            if (it == by_type.end()) continue;
            std::sort(it->second.begin(), it->second.end());
            extra_patterns += "                            Custom "
                            + fractal_type_to_string(t) + ": ";
            for (size_t i = 0; i < it->second.size(); i++) {
                extra_patterns += it->second[i];
                if (i + 1 < it->second.size()) extra_patterns += ", ";
            }
            extra_patterns += "\n";
        }
    }

    std::cout << help_text1 + extra_patterns + help_text2;
}

void cli_error(const std::string& message) {
    std::cerr << message << std::endl;
    cli_help();
    exit(1);
}

void cli_cast_to_num(const std::string& arg, int& dst, const std::string& flag) {
    try { dst = std::stoi(arg); }
    catch (...) { cli_error("Expected a integer type value for " + flag + "\nGot " + arg); }
}

void cli_cast_to_num(const std::string& arg, double& dst, const std::string& flag) {
    try { dst = std::stod(arg); }
    catch (...) { cli_error("Expected a double type value for " + flag + "\nGot " + arg); }
}

static void parse_size_arg(const std::string& arg, std::string& width, std::string& height) {
    std::string size_arg = arg;
    util::to_lowercase(size_arg);
    size_t sep = size_arg.find('x');
    if (sep == std::string::npos)       cli_error("Missing an \"x\" as separator for the dimensions");
    if (sep >= size_arg.size() - 1)     cli_error("Missing a height value");
    width  = size_arg.substr(0, sep);
    height = size_arg.substr(sep + 1);
}

static void parse_julia_c_arg(const std::string& arg, double& re, double& im) {
    size_t sep = arg.find(',');
    if (sep == std::string::npos) cli_error("--julia-c expects \"re,im\" (comma-separated)");
    cli_cast_to_num(arg.substr(0, sep), re, "--julia-c re");
    cli_cast_to_num(arg.substr(sep + 1), im, "--julia-c im");
}

void parse_cli_args(int argc, char* argv[], int& width, int& height,
                    std::string& pattern, std::string& theme,
                    int& max_iter, double& zoom_factor, bool& smooth,
                    double& julia_c_re, double& julia_c_im, bool& julia_c_set) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help") { cli_help(); exit(0); }
    }

    for (int i = 1; i < argc; ) {
        std::string arg = argv[i];
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);

        auto require_value = [&](const char* flag) {
            if (i + 1 >= argc) cli_error(std::string("Missing a value following ") + flag);
        };

        if (arg == "--size") {
            require_value("--size");
            std::string sw, sh;
            parse_size_arg(argv[i + 1], sw, sh);
            cli_cast_to_num(sw, width, "the width of --size");
            cli_cast_to_num(sh, height, "the height of --size");
            if (width < 100 || height < 100) { width = 800; height = 600; }
            i += 2;
        }
        else if (arg == "--pattern") {
            require_value("--pattern");
            pattern = argv[i + 1];
            std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
            if (goals.find(pattern) == goals.end())
                cli_error(pattern + " is not a valid --pattern value");
            i += 2;
        }
        else if (arg == "--max-iter") {
            require_value("--max-iter");
            cli_cast_to_num(argv[i + 1], max_iter, arg);
            i += 2;
        }
        else if (arg == "--zoom-factor") {
            require_value("--zoom-factor");
            cli_cast_to_num(argv[i + 1], zoom_factor, arg);
            i += 2;
        }
        else if (arg == "--smooth") {
            smooth = true;
            i += 1;
        }
        else if (arg == "--theme") {
            require_value("--theme");
            theme = argv[i + 1];
            i += 2;
        }
        else if (arg == "--julia-c") {
            require_value("--julia-c");
            parse_julia_c_arg(argv[i + 1], julia_c_re, julia_c_im);
            julia_c_set = true;
            i += 2;
        }
        else {
            cli_error(arg + " is not a valid flag!");
        }
    }
}
