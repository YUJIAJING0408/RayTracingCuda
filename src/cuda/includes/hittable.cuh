//
// Created by YUJIAJING on 25-7-28.
//

#ifndef HITTABLE_CUH
#define HITTABLE_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "raytracingCommon.cuh"

enum class shapeType { SPHERE, PLANE, TRIANGLE };

struct sphereData {
    point3 center;
    float radius;
    material material;
};

struct planeData {
    float y;
    material material;
};

struct shape {
    shapeType type;
    sphereData sphere;
    planeData plane;
};

__device__ bool hitSphere(const sphereData &s, const ray &r, const interval rayT, hitRecord &rec) {
    vec3 oc = s.center - r.GetOrigin();
    // r.GetDirection().print();
    float a = r.GetDirection().lengthSquared();
    float h = dot(r.GetDirection(), oc);
    float c = oc.lengthSquared() - s.radius * s.radius;

    float discriminant = h * h - a * c;
    // printf("a,h,c,d = %f,%f,%f,%f\n", a,h,c,discriminant);
    const float epsilon = 1e-8f;
    if (discriminant < epsilon) return false;
    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    // printf("a,h,c,root is %f,%f,%f,%f\n",a,h,c,root);
    if (!rayT.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!rayT.surrounds(root)) {
            return false;
        }
    }
    // printf("a,h,c,root is %f,%f,%f,%f\n",a,h,c,root);
    rec.t = root;
    rec.p = r.at(rec.t);
    rec.mat = s.material;
    vec3 outward_normal = (rec.p - s.center) / s.radius;
    rec.set_face_normal(r, outward_normal);
    return true;
}

__device__ bool hit(const shape s, const ray &r, const interval rayT, hitRecord &rec) {
    switch (s.type) {
        case shapeType::SPHERE: {
            return hitSphere(s.sphere, r, rayT, rec);
        }
        case shapeType::PLANE: {
            //
            return false;
        }
        default:
            printf("未知形状");
            return false;
    }
}

__device__ bool hitShapeList(shape *s, int worldSize, const ray &r, const interval rayT, hitRecord &rec) {
    // make sure s is on gpu mem
    hitRecord tempRec{
        point3(0, 0, 0),
        vec3(0, 0, 0),
        {
            materialType::LAMBERTIAN, {
                color(1.0f, 0.0f, 0.0f),
            }
        }
    };
    auto closest_so_far = rayT.max;
    bool hitAnything = false;
    for (int i = 0; i < worldSize; i++) {
        if (hit(s[i], r, interval(rayT.min, closest_so_far), tempRec)) {
            hitAnything = true;
            closest_so_far = tempRec.t;
            rec = tempRec;
        }
    }

    return hitAnything;
}


#endif //HITTABLE_CUH
