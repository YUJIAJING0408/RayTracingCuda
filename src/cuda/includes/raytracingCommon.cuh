//
// Created by YUJIAJING on 25-7-28.
//

#ifndef RAYTRACINGCOMMON_H
#define RAYTRACINGCOMMON_H
#define CUDALIMITSTACKSIZE 1024 * 128
#define MAXSPP 512
#define IMAGEWIDTH 400
#define MAXDEPTH 32
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <fstream>
#include <cstdlib>
#include <random>
#include <curand_kernel.h>

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

inline int randomInt(int min, int max) {
    return static_cast<int>(randomFloat(min, max + 1));
}

inline void progressBar(int current, int total, int bar_width) {
    float progress = static_cast<float>(current) / total;
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


// no err return false
__host__ bool isCudaError(cudaError_t err,const char * name,const int line) {
    if (err == cudaSuccess) return false;
    printf("CUDA ERROR (%s) is %s in %s at line-%d \n",cudaGetErrorName(err), cudaGetErrorString(err), name, line);
    return true;
};

// Common Headers
#include "vec3.cuh"
#include "color.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "aabb.cuh"
#include "material.cuh"
#include "hittable.cuh"
#include "cameraCuda.cuh"




#endif //RAYTRACINGCOMMON_H
