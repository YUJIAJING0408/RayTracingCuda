//
// Created by YUJIAJING on 25-7-28.
//

#ifndef INTERVAL_H
#define INTERVAL_H
#include "raytracingCommon.h"


class intervalCpu {
public:
    float min, max;

    intervalCpu() : min(+infinity), max(-infinity) {} // Default interval is empty

    intervalCpu(const float min, const float max) : min(min), max(max) {}

    float size() const {
        return max - min;
    }

    bool contains(const float x) const {
        return min <= x && x <= max;
    }

    bool surrounds(const float x) const {
        return min < x && x < max;
    }

    float clamp(float x) const {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const intervalCpu empty, universe;
};

const intervalCpu intervalCpu::empty    = intervalCpu(+infinity, -infinity);
const intervalCpu intervalCpu::universe = intervalCpu(-infinity, +infinity);


#endif //INTERVAL_H
