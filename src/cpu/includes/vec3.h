//
// Created by YUJIAJING on 25-7-30.
//

#ifndef VEC3_H
#define VEC3_H
#include <cmath>

class vec3cpu {
public:
    float element[3];
    vec3cpu() : element{0,0,0} {};
    vec3cpu(const float x, const float y, const float z):element{x,y,z} {}
    float x() const { return element[0]; }
    float y() const { return element[1]; }
    float z() const { return element[2]; }
    void print() const {
        printf("x: %f, y: %f, z: %f\n", element[0], element[1], element[2]);
    }
    vec3cpu operator-() const {
        return vec3cpu(-element[0], -element[1], -element[2]);
    }
    // +=
    vec3cpu& operator+=(const vec3cpu& v) {
        element[0] += v.element[0];
        element[1] += v.element[1];
        element[2] += v.element[2];
        return *this;
    }
    // -=
    vec3cpu& operator-=(const vec3cpu& v) {
        return *this += -v;
    }
    // *=
    vec3cpu& operator*=(const float s) {
        element[0] *= s;
        element[1] *= s;
        element[2] *= s;
        return *this;
    }
    // /=
    vec3cpu& operator/=(const float s) {
        return *this *= 1 / s;
    }
    float length() const {
        return sqrtf(lengthSquared());
    }
    float lengthSquared() const {
        return element[0] * element[0] + element[1] * element[1] + element[2] * element[2];
    }
    bool nearZero() const {
        float s = 1e-8f;
        return (fabsf(element[0]) < s) && (fabsf(element[1]) < s) && (fabsf(element[2]) < s);
    }
};

using point3cpu = vec3cpu;

inline vec3cpu operator+(const vec3cpu& l, const vec3cpu& r) {
    return vec3cpu(l.element[0]+r.element[0],l.element[1]+r.element[1],l.element[2]+r.element[2]);
}

inline vec3cpu operator-(const vec3cpu& l, const vec3cpu& r) {
    return vec3cpu(l.element[0]-r.element[0],l.element[1]-r.element[1],l.element[2]-r.element[2]);
}

inline vec3cpu operator*(const float s, const vec3cpu& v) {
    return vec3cpu(v.element[0]*s,v.element[1]*s,v.element[2]*s);
}

inline vec3cpu operator*(const vec3cpu& v,const float s) {
    return s*v;
}

inline vec3cpu operator*(const vec3cpu& l, const vec3cpu& r) {
    return vec3cpu(l.element[0]*r.element[0],l.element[1]*r.element[1],l.element[2]*r.element[2]);
}

inline vec3cpu operator/(const vec3cpu& v,const float s) {
    return (1/s)*v;
}

inline float dot(const vec3cpu &u, const vec3cpu &v) {
    return u.element[0] * v.element[0]
         + u.element[1] * v.element[1]
         + u.element[2] * v.element[2];
}

inline vec3cpu cross(const vec3cpu &u, const vec3cpu &v) {
    return vec3cpu(u.element[1] * v.element[2] - u.element[2] * v.element[1],
                u.element[2] * v.element[0] - u.element[0] * v.element[2],
                u.element[0] * v.element[1] - u.element[1] * v.element[0]);
}

inline vec3cpu unit_vector(const vec3cpu& v) {
    if (v.length() <= 0.0f) {
        return vec3cpu(0,0,0);
    }
    return v / v.length();
}

static vec3cpu random() {
    return vec3cpu(randomFloat(), randomFloat(), randomFloat());
}

static vec3cpu random(double min, double max) {
    return vec3cpu(randomFloat(min,max), randomFloat(min,max), randomFloat(min,max));
}
inline vec3cpu random_unit_vector() {
    while (true) {
        auto p = random(-1,1);
        auto lensq = p.lengthSquared();
        if (1e-32<lensq&&lensq <= 1)
            return p / sqrt(lensq);
    }
}

inline vec3cpu random_on_hemisphere(const vec3cpu& normal) {
    vec3cpu on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    return -on_unit_sphere;
}

inline vec3cpu reflect(const vec3cpu& v, const vec3cpu& n) {
    return v - 2*dot(v,n)*n;
}

inline vec3cpu refract(const vec3cpu& uv, const vec3cpu& n, float etai_over_etat) {
    auto cos_theta = std::fminf(dot(-uv, n), 1.0);
    vec3cpu r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3cpu r_out_parallel = -std::sqrtf(std::fabsf(1.0 - r_out_perp.lengthSquared())) * n;
    return r_out_perp + r_out_parallel;
}
#endif //VEC3_H
