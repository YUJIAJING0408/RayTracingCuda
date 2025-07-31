//
// Created by YUJIAJING on 25-7-30.
//

#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif



enum class materialType { LAMBERTIAN, METAL };
struct lambertianData {
    color albedo;
};

struct metalData {
    color albedo;
};

struct material {
    materialType type;
    lambertianData lambertian;
    metalData metal;
    // other
};


class hitRecord {
public:
    point3 p; // hit point
    vec3 normal; // hit point normal
    material mat;
    float t; // hit time
    bool front_face;
    // CUDA_CALLABLE hitRecord(): p(.0f, .0f, .0f), normal(.0f, .0f, .0f), mat(), t(infinity), front_face(false) {
    //     material m{
    //         materialType::LAMBERTIAN,
    //         {
    //             color(1.0f, 0.0f, 0.0f),
    //         }
    //     };
    //     mat = m;
    // };
    CUDA_CALLABLE void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.GetDirection(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


__device__ bool lambertianScatter(int idx, curandState *states,const lambertianData mat , const hitRecord& rec, color& attenuation, ray& scattered)
{
    vec3 scatterDirection = rec.normal + getRandomUnitVec3(idx,states);
    if (scatterDirection.nearZero()) {
        scatterDirection = rec.normal;
    }
    scattered = ray(rec.p, scatterDirection);
    attenuation = mat.albedo;
    return true;
}

__device__ bool metalScatter(const metalData mat ,const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered)
{
    vec3 reflected = reflect(r_in.GetDirection(), rec.normal);
    scattered = ray(rec.p, reflected);
    attenuation = mat.albedo;
    return true;
}

__device__ bool scatter(int idx, curandState *states,material mat,const ray& rayIn,const hitRecord& rec,color& attenuation,ray& rayOut) {
    // if (rec.mat.type!=materialType::LAMBERTIAN && rec.mat.type!=materialType::METAL) {
    //     printf("uk mat %d\n",rec.mat.type);
    // }
    // switch (mat.type) {
    //     case materialType::LAMBERTIAN: {
    //         lambertianScatter(idx,states,mat.lambertian,rec,attenuation,rayOut);
    //         // return true;
    //     }
    //     case materialType::METAL: {
    //         metalScatter(mat.metal,rayIn,rec,attenuation,rayOut);
    //         // return true;
    //     }
    //     default:
    //         printf("unknown material type\n");
    // }
    if (mat.type == materialType::METAL) {
        return metalScatter(mat.metal,rayIn,rec,attenuation,rayOut);
    }else {
        return lambertianScatter(idx,states,mat.lambertian,rec,attenuation,rayOut);
    }
}
#endif //MATERIAL_CUH
