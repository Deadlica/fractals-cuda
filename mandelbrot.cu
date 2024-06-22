#include <iostream>
#include <algorithm>
#include "palette.h"
#include <opencv4/opencv2/core/matx.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

const std::string WINDOW_NAME = "Mandelbrot Set";

struct coord {
  double x;
  double y;
};

struct goal {
  coord min;
  coord max;
};

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
    {-2.540158338, -4.9e-8},
    {-1.340156339, -0.000001548}
  }}
};

__device__ Color linear_interpolate(const Color& color1, const Color& color2, double t) {
    unsigned char r = static_cast<unsigned char>(color1.r + t * (color2.r - color1.r));
    unsigned char g = static_cast<unsigned char>(color1.g + t * (color2.g - color1.g));
    unsigned char b = static_cast<unsigned char>(color1.b + t * (color2.b - color1.b));
    return Color(r, g, b);
}

__global__ void mandelbrotKernel(Color *d_image, Color* PALETTE, int* palette_size, int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter, bool smooth = true) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    double x0 = x_min + idx * (x_max - x_min) / width;
    double y0 = y_min + idy * (y_max - y_min) / height;
    double x = 0.0, y = 0.0;
    int iter = 0;

    while (x*x + y*y <= 4.0 && iter < max_iter) {
        double xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }


    if (smooth) {
      double t;
      double iter_d;
      if (iter < max_iter) {
        double log_zn = logf(x*x + y*y) / 2.0f;
        double nu = logf(log_zn / logf(2.0f)) / logf(2.0f);
        iter_d = iter + 1 - nu;
        iter = static_cast<int>(floor(iter_d));
      }
      t = iter_d - iter;
      Color color1 = PALETTE[iter % *palette_size];
      Color color2 = PALETTE[(iter + 1) % *palette_size];
      Color color = linear_interpolate(color1, color2, t);
      d_image[idy * width + idx] = color;
    } else {
      d_image[idy * width + idx] = PALETTE[iter % *palette_size];
    }
}

void mandelbrot(Color *h_image, int width, int height, double x_min, double x_max, double y_min, double y_max, int max_iter) {
    Color *d_image;
    size_t imageSize = width * height * sizeof(Color);
    cudaMalloc(&d_image, imageSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    mandelbrotKernel<<<gridSize, blockSize>>>(d_image, PALETTE, PALETTE_SIZE, width, height, x_min, x_max, y_min, y_max, max_iter);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_image);
}

void displayImage(Color *h_image, int width, int height) {
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


void cli_help() {
    std::string help_text = R"(
Usage: mandelbrot [options]

Options:
  --size <value>          Set the width and height of the output image (default: 600)
  --pattern <name>        Choose a predefined pattern to zoom into (default: Mandelbrot)
                          Available patterns: flower, julia, seahorse, starfish, sun, tendrils, tree
  --max-iter <value>      Set the maximum number of iterations per pixel (default: 1500)
  --zoom-factor <value>   Set the zoom factor per iteration (default: 0.99)
  --help                  Display this help message

Examples:
  mandelbrot --pattern seahorse 
  mandelbrot --size 1000
  mandelbrot --pattern flower --zoom-factor 0.97
)";
  std::cout << help_text << std::endl;
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
  catch(...) {
    cli_error("Expected a integer type value for " + flag);
  }
}

void cli_cast_to_num(char* argv[], int i, double& dst, std::string flag) {
  try {
    dst = std::stod(argv[i]);
  }
  catch(...) {
    cli_error("Expected a double type value for " + flag);
  }
}

void parse_cli_args(int argc, char* argv[], int& width, int& height,
                    std::string& pattern, int& max_iter, double& zoom_factor) {
  for (int i = 1; i < argc; i++) {
    std::string arg = std::string(argv[i]);
    if (arg == "--help") {
      cli_help();
      exit(0);
    }
  }

  for (int i = 1; i < argc; i += 2) {
    std::string arg = std::string(argv[i]);
    if (arg == "--size") {
      if (i + 1 >= argc) cli_error("Missing a value following --size");
      cli_cast_to_num(argv, i + 1, width, arg);
    }
    else if (arg == "--pattern") {
      if (i + 1 >= argc) cli_error("Missing a value following --pattern");
      pattern = std::string(argv[i + 1]);
      std::transform(pattern.begin(), pattern.end(), pattern.begin(), ::tolower);
      if (goals.find(pattern) == goals.end()) cli_error(pattern + " is not a valid --pattern value");
    }
    else if (arg == "--max-iter") {
      if (i + 1 >= argc) cli_error("Missing a value following --max-iter");
      cli_cast_to_num(argv, i + 1, max_iter, arg);
    }
    else if (arg == "--zoom-factor") {
      if (i + 1 >= argc) cli_error("Missing a value following --zoom-factor");
      cli_cast_to_num(argv, i + 1, zoom_factor, arg);
    }
    else {
      cli_error(arg + "is not a valid flag!");
    }
  }
}

int main(int argc, char* argv[]) {
  int width = 600;
  int height = 600;
  double x_min = -2.0, x_max = 1.0;
  double y_min = -1.5, y_max = 1.5;

  std::string pattern = "";
  double x_goal_min = x_min;
  double x_goal_max = x_max;
  double y_goal_min = y_min;
  double y_goal_max = y_max;
  int max_iter = 1500;
  double zoom_factor = 0.94;
  parse_cli_args(argc, argv, width, height, pattern, max_iter, zoom_factor);
  Color *h_image = new Color[width * height];

  if (!pattern.empty()) {
    goal g = goals[pattern];
    x_goal_min = g.min.x;
    x_goal_max = g.max.x;
    y_goal_min = g.min.y;
    y_goal_max = g.max.y;
  }

  cv::namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

  double dx = std::abs(x_goal_min - x_goal_max);
  double dy = std::abs(y_goal_min - y_goal_max);
  bool stop_zoom = false;

  initialize_palette();

  while (true) {
      if (stop_zoom) break;
      mandelbrot(h_image, width, height, x_min, x_max, y_min, y_max, max_iter);
      displayImage(h_image, width, height);

      // Zoom in
      double x_center = (x_goal_min + x_goal_max) / 2;
      double y_center = (y_goal_min + y_goal_max) / 2;
      double new_width = (x_max - x_min) * zoom_factor;
      double new_height = (y_max - y_min) * zoom_factor;
      x_min = x_center - new_width / 2;
      x_max = x_center + new_width / 2;
      y_min = y_center - new_height / 2;
      y_max = y_center + new_height / 2;

      if (std::abs(x_min - x_max) <= dx &&
          std::abs(y_min - y_max) <= dy) {
        stop_zoom = true;
      }
  }

  while (cv::getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) >= 1) {
    int key = cv::waitKey(10);
    if (key == 27) { // ASCII value os 'Esc'
      break;
    }
  }

  cv::destroyWindow(WINDOW_NAME);
  delete[] h_image;
  free_palette();
  return 0;
}
