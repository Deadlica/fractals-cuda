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

extern std::unordered_map<std::string, goal> goals;
extern std::unordered_map<std::string, goal> custom_goals;

void init_custom_cli_patterns(const std::string& filename);
void cli_help();
void cli_error(const std::string& message);
void cli_cast_to_num(char* argv[], int i, int& dst, std::string flag);
void cli_cast_to_num(char* argv[], int i, double& dst, std::string flag);
void parse_cli_args(int argc, char* argv[], int& width, int& height,
                    std::string& pattern, std::string& theme,
                    int& max_iter, double& zoom_factor, bool& smooth);

#endif // CLI_H
