#include "cli.h"
#include "palette.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/matx.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

const std::string WINDOW_NAME = "Mandelbrot Set";

__device__ Color linear_interpolate(const Color& color1, const Color& color2, double t) {
    unsigned char r = static_cast<unsigned char>(color1.r + t * (color2.r - color1.r));
    unsigned char g = static_cast<unsigned char>(color1.g + t * (color2.g - color1.g));
    unsigned char b = static_cast<unsigned char>(color1.b + t * (color2.b - color1.b));
    return Color(r, g, b);
}

__global__ void mandelbrot_kernel(Color* d_image, Color* PALETTE,
                                 int* palette_size, int width, int height,
                                 double x_min, double x_max, double y_min,
                                 double y_max, int max_iter,
                                 bool smooth = true) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) {
        return;
    }

    double x0 = x_min + idx * (x_max - x_min) / width;
    double y0 = y_min + idy * (y_max - y_min) / height;
    double x = 0.0, y = 0.0;
    int iter = 0;

    while (x * x + y * y <= 4.0 && iter < max_iter) {
        double xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        iter++;
    }

    if (smooth) {
        double t;
        double iter_d;
        if (iter < max_iter) {
            double log_zn = logf(x * x + y * y) / 2.0f;
            double nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
            iter_d = iter + 1 - nu;
            iter = static_cast<int>(floor(iter_d));
        }
        t = iter_d - iter;
        Color color1 = PALETTE[iter % *palette_size];
        Color color2 = PALETTE[(iter + 1) % *palette_size];
        Color color = linear_interpolate(color1, color2, t);
        d_image[idy * width + idx] = color;
    }
    else {
        d_image[idy * width + idx] = PALETTE[iter % *palette_size];
    }
}

void mandelbrot(Color* h_image, int width, int height, double x_min,
                double x_max, double y_min, double y_max, int max_iter, bool smooth) {
    Color* d_image;
    size_t imageSize = width * height * sizeof(Color);
    cudaMalloc(&d_image, imageSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    mandelbrot_kernel<<<gridSize, blockSize>>>(d_image, PALETTE, PALETTE_SIZE,
                                               width, height, x_min, x_max, y_min,
                                               y_max, max_iter, smooth);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}

void display_image(Color* h_image, int width, int height) {
    cv::Mat img(height, width, CV_8UC3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color color = h_image[y * width + x];
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(color.b, color.r, color.g);
        }
    }
    cv::imshow(WINDOW_NAME, img);
    cv::waitKey(1);
}

void center_window(int window_width, int window_height) {
#ifdef _WIN32
    RECT desktop;
    const HWND hDesktop = GetDesktopWindow();
    GetWindowRect(hDesktop, &desktop);
    int screen_width = desktop.right;
    int screen_height = desktop.bottom;
#else
    int screen_width = 1920 / 2;
    int screen_height = 1080 / 2;
#endif
    int x = (screen_width - window_width) / 2;
    int y = (screen_height - window_height) / 2;
    cv::moveWindow(WINDOW_NAME, x, y);
}

int main(int argc, char* argv[]) {
    int width = 600;
    int height = 600;
    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;
    double x_goal_min = x_min;
    double x_goal_max = x_max;
    double y_goal_min = y_min;
    double y_goal_max = y_max;

    std::string pattern = "";
    int max_iter = 1500;
    double zoom_factor = 0.95;
    bool smooth = false;
    parse_cli_args(argc, argv, width, height, pattern, max_iter, zoom_factor, smooth);
    Color *h_image = new Color[width * height];

    if (!pattern.empty()) {
        goal g = goals[pattern];
        x_goal_min = g.min.x;
        x_goal_max = g.max.x;
        y_goal_min = g.min.y;
        y_goal_max = g.max.y;
    }

    cv::namedWindow(WINDOW_NAME);
    center_window(width, height);

    double dx = std::abs(x_goal_min - x_goal_max);
    double dy = std::abs(y_goal_min - y_goal_max);

    initialize_palette();

    while (true) {
        mandelbrot(h_image, width, height, x_min, x_max, y_min, y_max, max_iter, smooth);
        display_image(h_image, width, height);

        // Zoom in
        double x_center = (x_goal_min + x_goal_max) / 2;
        double y_center = (y_goal_min + y_goal_max) / 2;
        double new_width = (x_max - x_min) * zoom_factor;
        double new_height = (y_max - y_min) * zoom_factor;
        x_min = x_center - new_width / 2;
        x_max = x_center + new_width / 2;
        y_min = y_center - new_height / 2;
        y_max = y_center + new_height / 2;

        if (std::abs(x_min - x_max) <= dx && std::abs(y_min - y_max) <= dy) {
            break;
        }
    }

    while (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) >= 1) {
        int key = cv::waitKey(10);
        if (key == 27) { // ASCII value of 'Esc'
            break;
        }
    }

    cv::destroyWindow(WINDOW_NAME);
    delete[] h_image;
    free_palette();
    return 0;
}
