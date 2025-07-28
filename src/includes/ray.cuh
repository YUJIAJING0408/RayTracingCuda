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
    CUDA_CALLABLE ray(vec3 origin, vec3 direction): origin(origin), direction(direction) {}
    const point3& GetOrigin() const { return origin; }
    const vec3& GetDirection() const { return direction; }
    CUDA_CALLABLE point3 at(float t) const { return origin + t * direction; }

private:
    point3 origin;
    vec3 direction;
};


#endif //RAY_CUH
