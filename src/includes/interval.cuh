//
// Created by YUJIAJING on 25-7-28.
//

#ifndef INTERVAL_CUH
#define INTERVAL_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "raytracingCommon.h"
// #include "raytracingCommon.h"


class interval {
public:
    float min, max;

    CUDA_CALLABLE interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    CUDA_CALLABLE interval(const float min, const float max) : min(min), max(max) {}

    CUDA_CALLABLE double size() const {
        return max - min;
    }

    CUDA_CALLABLE bool contains(const float x) const {
        return min <= x && x <= max;
    }

    CUDA_CALLABLE bool surrounds(const float x) const {
        return min < x && x < max;
    }

    CUDA_CALLABLE float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

const interval interval::empty    = interval(+infinity, -infinity);
const interval interval::universe = interval(-infinity, +infinity);


#endif //INTERVAL_CUH
