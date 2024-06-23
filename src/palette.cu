#include "palette.h"
#include <vector>

std::vector<Color> v_palette = {
    Color(0, 0, 0), Color(0, 0, 170), Color(0, 170, 0), Color(0, 170, 170),
    Color(170, 0, 0), Color(170, 0, 170), Color(170, 85, 0), Color(170, 170, 170),
    Color(85, 85, 85), Color(85, 85, 255), Color(85, 255, 85), Color(85, 255, 255),
    Color(255, 85, 85), Color(255, 85, 255), Color(255, 255, 85), Color(255, 255, 255),
    Color(0, 0, 0), Color(20, 20, 20), Color(32, 32, 32), Color(44, 44, 44),
    Color(56, 56, 56), Color(68, 68, 68), Color(80, 80, 80), Color(96, 96, 96),
    Color(112, 112, 112), Color(128, 128, 128), Color(144, 144, 144), Color(160, 160, 160),
    Color(176, 176, 176), Color(192, 192, 192), Color(208, 208, 208), Color(224, 224, 224),
    Color(0, 0, 255), Color(64, 0, 255), Color(125, 0, 255), Color(188, 0, 255),
    Color(255, 0, 255), Color(255, 0, 188), Color(255, 0, 125), Color(255, 0, 64),
    Color(255, 0, 0), Color(255, 64, 0), Color(255, 125, 0), Color(255, 188, 0),
    Color(255, 255, 0), Color(188, 255, 0), Color(125, 255, 0), Color(64, 255, 0),
    Color(0, 255, 0), Color(0, 255, 64), Color(0, 255, 125), Color(0, 255, 188),
    Color(0, 255, 255), Color(0, 188, 255), Color(0, 125, 255), Color(0, 64, 255),
    Color(125, 125, 255), Color(156, 125, 255), Color(188, 125, 255), Color(220, 125, 255),
    Color(255, 125, 255), Color(255, 125, 220), Color(255, 125, 188), Color(255, 125, 156),
    Color(255, 125, 125), Color(255, 156, 125), Color(255, 188, 125), Color(255, 220, 125),
    Color(255, 255, 125), Color(220, 255, 125), Color(188, 255, 125), Color(156, 255, 125),
    Color(125, 255, 125), Color(125, 255, 156), Color(125, 255, 188), Color(125, 255, 220),
    Color(125, 255, 255), Color(125, 220, 255), Color(125, 188, 255), Color(125, 156, 255),
    Color(188, 188, 255), Color(204, 188, 255), Color(220, 188, 255), Color(237, 188, 255),
    Color(255, 188, 255), Color(255, 188, 237), Color(255, 188, 220), Color(255, 188, 204),
    Color(255, 188, 188), Color(255, 204, 188), Color(255, 220, 188), Color(255, 237, 188),
    Color(255, 255, 188), Color(237, 255, 188), Color(220, 255, 188), Color(204, 255, 188),
    Color(188, 255, 188), Color(188, 255, 204), Color(188, 255, 220), Color(188, 255, 237),
    Color(188, 255, 255), Color(188, 237, 255), Color(188, 220, 255), Color(188, 204, 255),
    Color(0, 0, 113), Color(28, 0, 113), Color(56, 0, 113), Color(85, 0, 113),
    Color(113, 0, 113), Color(113, 0, 85), Color(113, 0, 56), Color(113, 0, 28),
    Color(113, 0, 0), Color(113, 28, 0), Color(113, 56, 0), Color(113, 85, 0),
    Color(113, 113, 0), Color(85, 113, 0), Color(56, 113, 0), Color(28, 113, 0),
    Color(0, 113, 0), Color(0, 113, 28), Color(0, 113, 56), Color(0, 113, 85),
    Color(0, 113, 113), Color(0, 85, 113), Color(0, 56, 113), Color(0, 28, 113),
    Color(56, 56, 113), Color(68, 56, 113), Color(85, 56, 113), Color(96, 56, 113),
    Color(113, 56, 113), Color(113, 56, 96), Color(113, 56, 85), Color(113, 56, 68),
    Color(113, 56, 56), Color(113, 68, 56), Color(113, 85, 56), Color(113, 96, 56),
    Color(113, 113, 56), Color(96, 113, 56), Color(85, 113, 56), Color(68, 113, 56),
    Color(56, 113, 56), Color(56, 113, 68), Color(56, 113, 85), Color(56, 113, 96),
    Color(56, 113, 113), Color(56, 96, 113), Color(56, 85, 113), Color(56, 68, 113),
    Color(80, 80, 113), Color(88, 80, 113), Color(96, 80, 113), Color(105, 80, 113),
    Color(113, 80, 113), Color(113, 80, 105), Color(113, 80, 96), Color(113, 80, 88),
    Color(113, 80, 80), Color(113, 88, 80), Color(113, 96, 80), Color(113, 105, 80),
    Color(113, 113, 80), Color(105, 113, 80), Color(96, 113, 80), Color(88, 113, 80),
    Color(80, 113, 80), Color(80, 113, 88), Color(80, 113, 96), Color(80, 113, 105),
    Color(80, 113, 113), Color(80, 105, 113), Color(80, 96, 113), Color(80, 88, 113),
    Color(0, 0, 64), Color(16, 0, 64), Color(32, 0, 64), Color(48, 0, 64),
    Color(64, 0, 64), Color(64, 0, 48), Color(64, 0, 32), Color(64, 0, 16),
    Color(64, 0, 0), Color(64, 16, 0), Color(64, 32, 0), Color(64, 48, 0),
    Color(64, 64, 0), Color(48, 64, 0), Color(32, 64, 0), Color(16, 64, 0),
    Color(0, 64, 0), Color(0, 64, 16), Color(0, 64, 32), Color(0, 64, 48),
    Color(0, 64, 64), Color(0, 48, 64), Color(0, 32, 64), Color(0, 16, 64),
    Color(32, 32, 64), Color(40, 32, 64), Color(48, 32, 64), Color(56, 32, 64),
    Color(64, 32, 64), Color(64, 32, 56), Color(64, 32, 48), Color(64, 32, 40),
    Color(64, 32, 32), Color(64, 40, 32), Color(64, 48, 32), Color(64, 56, 32),
    Color(64, 64, 32), Color(56, 64, 32), Color(48, 64, 32), Color(40, 64, 32),
    Color(32, 64, 32), Color(32, 64, 40), Color(32, 64, 48), Color(32, 64, 56),
    Color(32, 64, 64), Color(32, 56, 64), Color(32, 48, 64), Color(32, 40, 64),
    Color(44, 44, 64), Color(48, 44, 64), Color(56, 44, 64), Color(60, 44, 64),
    Color(64, 44, 64), Color(64, 44, 60), Color(64, 44, 56), Color(64, 44, 48),
    Color(64, 44, 44), Color(64, 48, 44), Color(64, 56, 44), Color(64, 60, 44),
    Color(64, 64, 44), Color(60, 64, 44), Color(56, 64, 44), Color(48, 64, 44),
    Color(44, 64, 44), Color(44, 64, 48), Color(44, 64, 56), Color(44, 64, 60),
    Color(44, 64, 64), Color(44, 60, 64), Color(44, 56, 64), Color(44, 48, 64),
    Color(0, 0, 0), Color(0, 0, 0), Color(0, 0, 0), Color(0, 0, 0)
};

int v_palette_size = v_palette.size();


Color* PALETTE = nullptr;
int* PALETTE_SIZE = nullptr;

void initialize_palette() {
    size_t palette_size = v_palette.size() * sizeof(Color);
    size_t int_size = sizeof(v_palette_size);

    cudaMalloc(&PALETTE, palette_size);
    cudaMalloc(&PALETTE_SIZE, int_size);

    cudaMemcpy(PALETTE, v_palette.data(), palette_size, cudaMemcpyHostToDevice);
    cudaMemcpy(PALETTE_SIZE, &v_palette_size, int_size, cudaMemcpyHostToDevice);
}

void free_palette() {
    cudaFree(PALETTE);
}
