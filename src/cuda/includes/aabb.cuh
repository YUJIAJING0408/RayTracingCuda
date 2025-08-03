//
// Created by YUJIAJING on 25-8-2.
//

#ifndef AABB_CUH
#define AABB_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "interval.cuh"
class AABB {
public:
    interval x, y, z; // AABB x y z
    CUDA_CALLABLE AABB() {
    }; // default AABB is empty
    CUDA_CALLABLE AABB(const interval &x, const interval &y, const interval &z): x(x), y(y), z(z) {
    };
    CUDA_CALLABLE AABB(const point3 &a, const point3 &b) {
        x = (a.x() <= b.x()) ? interval(a.x(), b.x()) : interval(b.x(), a.x());
        y = (a.y() <= b.y()) ? interval(a.y(), b.y()) : interval(b.y(), a.y());
        z = (a.z() <= b.z()) ? interval(a.z(), b.z()) : interval(b.z(), a.z());
    }

    CUDA_CALLABLE AABB(const AABB &a, const AABB &b) {
        x = interval(a.x, b.x);
        y = interval(a.y, b.y);
        z = interval(a.z, b.z);
    }

    const CUDA_CALLABLE interval &axisInterval(int axis) const {
        if (axis == 0) return x;
        if (axis == 1) return y;
        if (axis == 2) return z;
        return x;
    }

    CUDA_CALLABLE bool hit(const ray &r, interval rayT) const {
        const point3 rayOrigin = r.GetOrigin();
        auto rayDirection = r.GetDirection();
        for (int axis = 0; axis < 3; axis++) {
            const interval &ax = axisInterval(axis);
            const float adinv = 1.f / rayDirection[axis];

            auto t0 = (ax.min - rayOrigin[axis]) * adinv;
            auto t1 = (ax.max - rayOrigin[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > rayT.min) rayT.min = t0;
                if (t1 < rayT.max) rayT.max = t1;
            } else {
                if (t1 > rayT.min) rayT.min = t1;
                if (t0 < rayT.max) rayT.max = t0;
            }
            if (rayT.max <= rayT.min) return false;
        }
        return true;
    }
};
#endif //AABB_CUH
