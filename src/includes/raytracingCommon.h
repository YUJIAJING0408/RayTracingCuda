//
// Created by YUJIAJING on 25-7-28.
//

#ifndef RAYTRACINGCOMMON_H
#define RAYTRACINGCOMMON_H
#define MAXSPP 4096
#define IMAGEWIDTH 1024
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <fstream>
#include <cstdlib>
#include <random>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::cin;
using std::cout;

// Constants

constexpr float infinity = std::numeric_limits<double>::infinity();
constexpr float pi = 3.1415926535897932385;

// Utility Functions
inline float degrees2Radians(float degrees) {
    return degrees * pi / 180.0f;
}

inline float randomFloat() {
    // Returns a random real in [0,1).
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float randomFloat(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max-min)*randomFloat();
}

inline void progressBar(int current, int total, int bar_width) {
    float progress = (float)current / total;
    int pos = bar_width * progress;

    printf("[");
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100.0));

    printf("\r");
    fflush(stdout);

    if (current == total) printf("\n");
}

// Common Headers
#include "vec3.cuh"
#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "hittableList.cuh"
#include "shpere.cuh"
#include "cameraCuda.cuh"
#include "camera.cuh"

#endif //RAYTRACINGCOMMON_H
