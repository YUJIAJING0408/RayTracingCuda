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

class hitRecord {
public:
    point3 p; // hit point
    vec3 normal; // hit point normal
    float t; // hit time
    bool front_face;
    CUDA_CALLABLE hitRecord():p(.0f,.0f,.0f),normal(.0f,.0f,.0f),t(infinity),front_face(false){};
    CUDA_CALLABLE void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.GetDirection(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable{
    public:
    virtual ~hittable() = default;
    virtual CUDA_CALLABLE bool hit(const ray& r, interval rayT, hitRecord& rec) const = 0;
};

#endif //HITTABLE_CUH
