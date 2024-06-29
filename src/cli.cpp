#include "cli.h"
#include <algorithm>
#include <iostream>
#include <fstream>

std::unordered_map<std::string, goal> goals = {
    {"flower", {
        {-1.999985885, 2e-9},
        {-1.999985879, -3e-9}
    }},
    {"julia", {
        {-1.768779320, -0.001738521},
        {-1.768778317, -0.001739281}
    }},
    {"seahorse", {
        {-0.750555615, -0.121803013},
        {-0.736462413, -0.132368505}
    }},
    {"starfish", {
        {-0.375652034, 0.661031194},
        {-0.372352113, 0.658557285}
    }},
    {"sun", {
        {-0.776606539, -0.136630553},
        {-0.776579121, -0.136651108}
    }},
    {"tendrils", {
        {-0.226267721, 1.116175247},
        {-0.226265572, 1.116173636}
    }},
    {"tree", {
        {-1.940158339, -4.9e-8},
        {-1.940156342, -0.000001548}
    }}
};

std::unordered_map<std::string, goal> custom_goals;

void init_custom_cli_patterns(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return;
    }
    double x_min, x_max, y_min, y_max;
    std::string pattern_name;
    while (!file.eof()) {
        file >> pattern_name >> x_min >> y_min >> x_max >> y_max;
        if (pattern_name.empty()) continue;

        std::pair<std::string, goal> pattern = {pattern_name, {
                {x_min, y_min},
                {x_max, y_max}
        }};
        custom_goals.emplace(pattern);
        goals.emplace(pattern);
    }
    file.close();
}

void cli_help() {
    std::string help_text1 = R"(
Usage: mandelbrot [options]

Options:
    --size <value>          Set the width and height of the output image (default: 600)
    --pattern <name>        Choose a predefined pattern to zoom into (default: Mandelbrot)
                            Available patterns: flower, julia, seahorse, starfish, sun, tendrils, tree
)";
    std::string help_text2 = R"(
    --max-iter <value>      Set the maximum number of iterations per pixel (default: 1500)
    --zoom-factor <value>   Set the zoom factor per iteration (default: 0.99)
    --smooth                Enables smoothing, reduces color bands
    --theme                 Use a custom defined color theme.
                            the theme should be placed under 'themes/' and with the '.mt' extension
    --help                  Display this help message

Examples:
    mandelbrot --pattern seahorse --smooth
    mandelbrot --size 1000
    mandelbrot --pattern flower --zoom-factor 0.97
)";

    std::string extra_patterns;
    if (!custom_goals.empty()) {
        extra_patterns = R"(                            Custom patterns: )";
        auto it = custom_goals.begin();
        auto last = custom_goals.end();
        for (; it != last; it++) {
            extra_patterns += it->first;
            if (std::distance(it, last) > 1) {
                extra_patterns += ", ";
            }
            else {
                extra_patterns += "\n";
            }
        }
    }

    std::cout << help_text1 + extra_patterns + help_text2;
}

void cli_error(const std::string& message) {
    std::cout << message << std::endl;
    cli_help();
    exit(0);
}

void cli_cast_to_num(char* argv[], int i, int& dst, std::string flag) {
    try {
        dst = std::stoi(argv[i]);
    }
    catch (...) {
        cli_error("Expected a integer type value for " + flag);
    }
}

void cli_cast_to_num(char* argv[], int i, double& dst, std::string flag) {
    try {
        dst = std::stod(argv[i]);
    }
    catch (...) {
        cli_error("Expected a double type value for " + flag);
    }
}

void parse_cli_args(int argc, char* argv[], int& width, int& height,
                    std::string& pattern, std::string& theme,
                    int& max_iter, double& zoom_factor, bool& smooth) {
    for (int i = 1; i < argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "--help") {
            cli_help();
            exit(0);
        }
    }

    for (int i = 1; i < argc; i += 2) {
        std::string arg = std::string(argv[i]);
        std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
        if (arg == "--size") {
            if (i + 1 >= argc) {
                cli_error("Missing a value following --size");
            }
            cli_cast_to_num(argv, i + 1, width, arg);
        }
        else if (arg == "--pattern") {
            if (i + 1 >= argc) {
                cli_error("Missing a value following --pattern");
            }
            pattern = std::string(argv[i + 1]);
            std::transform(pattern.begin(), pattern.end(), pattern.begin(),
                           ::tolower);
            if (goals.find(pattern) == goals.end()) {
                cli_error(pattern + " is not a valid --pattern value");
            }
        }
        else if (arg == "--max-iter") {
            if (i + 1 >= argc) {
                cli_error("Missing a value following --max-iter");
            }
            cli_cast_to_num(argv, i + 1, max_iter, arg);
        }
        else if (arg == "--zoom-factor") {
            if (i + 1 >= argc) {
                cli_error("Missing a value following --zoom-factor");
            }
            cli_cast_to_num(argv, i + 1, zoom_factor, arg);
        }
        else if (arg == "--smooth") {
            smooth = true;
            i--;
        }
        else if (arg == "--theme") {
            if (i + 1 >= argc) {
                cli_error("Missing a value following --theme");
            }
            theme = std::string(argv[i + 1]);
        }
        else {
            cli_error(arg + "is not a valid flag!");
        }
    }
}
