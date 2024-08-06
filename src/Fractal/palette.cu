// Project
#include <Fractal/palette.cuh>
#include <Util/util.h>

// std
#include <vector>
#include <fstream>
#include <iostream>

// Default theme
std::vector<Color> v_palette = {
    Color(66, 30, 15),
    Color(25, 7, 26),
    Color(9, 1, 47),
    Color(4, 4, 73),
    Color(0, 7, 100),
    Color(12, 44, 138),
    Color(24, 82, 177),
    Color(57, 125, 209),
    Color(134, 181, 229),
    Color(211, 236, 248),
    Color(241, 233, 191),
    Color(248, 201, 95),
    Color(255, 170, 0),
    Color(204, 128, 0),
    Color(153, 87, 0),
    Color(106, 52, 3)
};

int v_palette_size = v_palette.size();


Color* PALETTE = nullptr;
int* PALETTE_SIZE = nullptr;

std::string get_theme_path(const std::string& theme) {
    std::string path = theme;
    if (!util::starts_with(theme, "themes/")) {
        path = "themes/" + theme;
    }
    if (!util::ends_with(theme, ".mbt")) {
        path += ".mbt";
    }

    return path;
}

void load_color_theme(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Could not find file: " << path << std::endl << "Using default." << std::endl;
        return;
    }

    std::vector<Color> temp_palette;
    
    std::string line;
    while (std::getline(file, line)) {
        size_t comma1 = line.find(",");
        if (comma1 == std::string::npos) {
            std::cout << "Invalid theme: " << path << std::endl << "Using default theme." << std::endl;
            file.close();
            return;
        }

        size_t comma2 = line.find(",", comma1 + 1);
        if (comma2 == std::string::npos) {
            std::cout << "Invalid theme: " << path << std::endl << "Using default theme." << std::endl;
            file.close();
            return;
        }

        try {
            int r = std::stoi(line.substr(0, comma1));
            int g = std::stoi(line.substr(comma1 + 1, comma2 - comma1 - 1));
            int b = std::stoi(line.substr(comma2 + 1));

            temp_palette.push_back(Color{static_cast<unsigned char>(r),
                                         static_cast<unsigned char>(g),
                                         static_cast<unsigned char>(b)});
        } catch (const std::invalid_argument& e) {
            std::cout << "Invalid value in theme: " << path << std::endl << "Using default theme." << std::endl;
            file.close();
            return;
        }
    }

    v_palette = temp_palette;
    v_palette_size = temp_palette.size();
}

void initialize_palette(const std::string& theme) {
    if (!theme.empty()) {
        std::string file_name = get_theme_path(theme);
        load_color_theme(file_name);
    }

    size_t palette_size = v_palette.size() * sizeof(Color);
    size_t int_size = sizeof(v_palette_size);

    cudaMalloc(&PALETTE, palette_size);
    cudaMalloc(&PALETTE_SIZE, int_size);

    cudaMemcpy(PALETTE, v_palette.data(), palette_size, cudaMemcpyHostToDevice);
    cudaMemcpy(PALETTE_SIZE, &v_palette_size, int_size, cudaMemcpyHostToDevice);
}

void free_palette() {
    cudaFree(PALETTE);
    cudaFree(PALETTE_SIZE);
}
