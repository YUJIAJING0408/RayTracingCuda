//
// Created by YUJIAJING on 25-7-28.
//

#ifndef VEC3_CUH
#define VEC3_CUH
#include <cmath>


#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class vec3 {
public:
    float element[3];
    CUDA_CALLABLE vec3() : element{0,0,0} {};
    CUDA_CALLABLE vec3(const float x, const float y, const float z):element{x,y,z} {}
    CUDA_CALLABLE float x() const { return element[0]; }
    CUDA_CALLABLE float y() const { return element[1]; }
    CUDA_CALLABLE float z() const { return element[2]; }
    CUDA_CALLABLE void print() const {
        printf("x: %f, y: %f, z: %f\n", element[0], element[1], element[2]);
    }
    CUDA_CALLABLE vec3 operator-() const {
        return vec3(-element[0], -element[1], -element[2]);
    }
    // +=
    CUDA_CALLABLE vec3& operator+=(const vec3& v) {
        element[0] += v.element[0];
        element[1] += v.element[1];
        element[2] += v.element[2];
        return *this;
    }
    // -=
    CUDA_CALLABLE vec3& operator-=(const vec3& v) {
        return *this += -v;
    }
    // *=
    CUDA_CALLABLE vec3& operator*=(const float s) {
        element[0] *= s;
        element[1] *= s;
        element[2] *= s;
        return *this;
    }
    // /=
    CUDA_CALLABLE vec3& operator/=(const float s) {
        return *this *= 1 / s;
    }
    CUDA_CALLABLE float length() const {
        return sqrtf(lengthSquared());
    }
    CUDA_CALLABLE float lengthSquared() const {
        return element[0] * element[0] + element[1] * element[1] + element[2] * element[2];
    }
};

using point3 = vec3;

CUDA_CALLABLE inline vec3 operator+(const vec3& l, const vec3& r) {
    return vec3(l.element[0]+r.element[0],l.element[1]+r.element[1],l.element[2]+r.element[2]);
}

CUDA_CALLABLE inline vec3 operator-(const vec3& l, const vec3& r) {
    return vec3(l.element[0]-r.element[0],l.element[1]-r.element[1],l.element[2]-r.element[2]);
}

CUDA_CALLABLE inline vec3 operator*(const float s, const vec3& v) {
    return vec3(v.element[0]*s,v.element[1]*s,v.element[2]*s);
}

CUDA_CALLABLE inline vec3 operator*(const vec3& v,const float s) {
    return s*v;
}

CUDA_CALLABLE inline vec3 operator*(const vec3& l, const vec3& r) {
    return vec3(l.element[0]*r.element[0],l.element[1]*r.element[1],l.element[2]*r.element[2]);
}

CUDA_CALLABLE inline vec3 operator/(const vec3& v,const float s) {
    return (1/s)*v;
}

CUDA_CALLABLE inline float dot(const vec3 &u, const vec3 &v) {
    return u.element[0] * v.element[0]
         + u.element[1] * v.element[1]
         + u.element[2] * v.element[2];
}

CUDA_CALLABLE inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.element[1] * v.element[2] - u.element[2] * v.element[1],
                u.element[2] * v.element[0] - u.element[0] * v.element[2],
                u.element[0] * v.element[1] - u.element[1] * v.element[0]);
}

CUDA_CALLABLE inline vec3 unit_vector(const vec3& v) {
    if (v.length() <= 0.0f) {
        return vec3(0,0,0);
    }
    return v / v.length();
}

#endif //VEC3_CUH