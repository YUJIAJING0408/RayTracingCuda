//
// Created by YUJIAJING on 25-7-28.
//

#ifndef RAYTRACINGCOMMON_H
#define RAYTRACINGCOMMON_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <fstream>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::cin;
using std::cout;

// Constants

constexpr float infinity = std::numeric_limits<double>::infinity();
constexpr float pi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

// Common Headers
#include "vec3.cuh"
#include "interval.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hittable.cuh"
#include "hittableList.cuh"
#include "shpere.cuh"
#include "camera.cuh"


#endif //RAYTRACINGCOMMON_H
