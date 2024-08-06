#ifndef CLI_H
#define CLI_H

#include <string>
#include <unordered_map>

struct coord {
    double x;
    double y;
};

struct goal {
    coord min;
    coord max;
};

struct size_arg {
    size_t width;
    size_t height;
};

extern std::unordered_map<std::string, goal> goals;
extern std::unordered_map<std::string, goal> custom_goals;

void init_custom_cli_patterns(const std::string& filename);
void cli_help();
void cli_error(const std::string& message);
void cli_cast_to_num(const std::string& arg, int& dst, const std::string& flag);
void cli_cast_to_num(const std::string& arg, double& dst, const std::string& flag);
void parse_size_arg(const std::string& size);
void parse_cli_args(int argc, char* argv[], int& width, int& height,
                    std::string& pattern, std::string& theme,
                    int& max_iter, double& zoom_factor, bool& smooth);

#endif // CLI_H
