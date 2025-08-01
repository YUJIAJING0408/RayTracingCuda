//
// Created by YUJIAJING on 25-7-28.
//

#ifndef RAY_CUH
#define RAY_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#include "vec3.cuh"

class ray {
public:
    CUDA_CALLABLE ray(): origin(0,0,0), direction(0,0,-1) {}
    CUDA_CALLABLE ray(vec3 origin, vec3 direction): origin(origin), direction(direction),tm(0.f) {}
    CUDA_CALLABLE ray(vec3 origin, vec3 direction, float time): origin(origin), direction(direction),tm(time) {}
    const CUDA_CALLABLE point3& GetOrigin() const { return origin; }
    const CUDA_CALLABLE vec3& GetDirection() const { return direction; }
    CUDA_CALLABLE point3 at(float t) const { return origin + t * direction; }
    const CUDA_CALLABLE float getTime() const {return tm;}

private:
    point3 origin;
    vec3 direction;
    float tm;
};


#endif //RAY_CUH
