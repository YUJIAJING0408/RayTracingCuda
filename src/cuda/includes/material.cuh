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
#include "vec3.cuh"

const enum textureType {
    CHECKERBOARD,
    SOLIDCOLOR
};

struct  solidColor {
public:
    solidColor() :albedo(color(0.5f,0.5f,0.5f)){}
    solidColor(const color& albedo) : albedo(albedo) {}
    __device__ color value(float u, float v, const point3& p) {
        return albedo;
    }
    color albedo;
};

struct  checkerBoard {
public:
    checkerBoard():invScale(1.0f),even(solidColor(color(1.f,1.0f,1.0f))),odd(color()){}
    checkerBoard(const float is,const solidColor& even,const solidColor& odd):invScale(is),even(even),odd(odd){}
    checkerBoard(const float is,const color& even,const color& odd):invScale(is),even(solidColor(even)),odd(solidColor(odd)){}
    checkerBoard(color c1,color c2):invScale(1.f),even(solidColor(c1)),odd(solidColor(c2)){}
    __device__ color value(float u, float v, const point3& p) {
        auto xInteger = int(std::floor(invScale * p.x()));
        auto yInteger = int(std::floor(invScale * p.y()));
        auto zInteger = int(std::floor(invScale * p.z()));
        bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;
        return isEven ? even.value(u, v, p) : odd.value(u, v, p);
    }
    float invScale;
    solidColor even;
    solidColor odd;
};

struct  texture {
public:
    texture():type(textureType::CHECKERBOARD),checkerboard(checkerBoard(1.0f,color(1.f,1.f,1.f),color(0.f,0.f,0.f))),solidColor(color(0.5f,0.5f,0.5f)){}
    texture(color c):type(SOLIDCOLOR),checkerboard(),solidColor(c){}
    texture(color c1,color c2):type(CHECKERBOARD),checkerboard(c1,c2),solidColor() {

    }
    texture(checkerBoard c):type(CHECKERBOARD),checkerboard(c),solidColor(){}
    __device__ color value(float u,float v,const point3& p) {
        if (type==textureType::CHECKERBOARD) {
            return  checkerboard.value(u,v,p);
        }else if (type==SOLIDCOLOR) {
            return solidColor.value(u,v,p);
        }
        return color(0.f,0.f,0.f);
    }
    textureType type;
    checkerBoard checkerboard;
    solidColor solidColor;
};

CUDA_CALLABLE texture makeSolidTexture(color c) {
    return texture(c);
}

CUDA_CALLABLE texture makeCheckerBoardTexture(const color c1,const color c2) {
    return texture(c1,c2);
}



enum class materialType { LAMBERTIAN, METAL ,DIELECTRIC};
struct lambertianData {
    texture albedoTexture;
};

struct metalData {
    color albedo;
    float fuzz;
};

struct dielectricData {
    float refractiveIndex;
};

struct material {
    materialType type;
    lambertianData lambertian;
    metalData metal;
    dielectricData dielectric;
    // other
};

__host__ material& makeLambertian(const color& albedo) {
    material m = {
        .type = materialType::LAMBERTIAN,
        .lambertian = {
            .albedoTexture = makeSolidTexture(albedo),
        }
    };
    return m;
}

__host__ material& makeLambertianCheckerBoard(const color& c1,const color& c2) {
    material m = {
        .type = materialType::LAMBERTIAN,
        .lambertian = {
            .albedoTexture = makeCheckerBoardTexture(c1,c2),
        }
    };
    return m;
}

__host__ material& makeMetal(const color albedo,const float fuzz) {
    material m = {
        .type = materialType::METAL,
        .metal = {
            .albedo = albedo,
            .fuzz = fuzz
        }
    };
    return m;
}

__host__ material& makeDielectric(const float refractiveIndex) {
    material m = {
        .type = materialType::DIELECTRIC,
        .dielectric = {
            .refractiveIndex = refractiveIndex
        }
    };
    return m;
}

class hitRecord {
public:
    point3 p; // hit point
    vec3 normal; // hit point normal
    material mat;
    float t; // hit time
    float u;
    float v;
    bool front_face;
    CUDA_CALLABLE void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.GetDirection(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


__device__ bool lambertianScatter(int idx, curandState *states,const lambertianData mat ,const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered)
{
    vec3 scatterDirection = rec.normal + getRandomUnitVec3(idx,states);
    if (scatterDirection.nearZero()) {
        scatterDirection = rec.normal;
    }
    scattered = ray(rec.p, scatterDirection,r_in.getTime());
    // if (mat.albedoTexture.type==textureType::SOLIDCOLOR){
    //     attenuation = mat.albedoTexture.solidColor.albedo;
    //     return true;
    // }
    texture t = mat.albedoTexture;
    attenuation = t.value(rec.u,rec.v,rec.p);
    return true;
}

__device__ bool metalScatter(int idx, curandState *states,const metalData mat ,const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered)
{
    vec3 reflected = reflect(r_in.GetDirection(), rec.normal);
    reflected = unit_vector(reflected) + getRandomUnitVec3(idx,states) * mat.fuzz;
    scattered = ray(rec.p, reflected,r_in.getTime());
    attenuation = mat.albedo;
    return (dot(scattered.GetDirection(), rec.normal) > 0);
}

__device__ float reflectance(float cosine,float refractionIndex) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
    r0 = r0*r0;
    return r0 + (1.0f - r0)*powf((1.0f - cosine),5);
}

__device__ bool dielectricScatter(int idx, curandState *states,const dielectricData mat ,const ray& r_in, const hitRecord& rec, color& attenuation, ray& scattered) {
    attenuation = color(1.0f, 1.0f, 1.0f);
    float ri =rec.front_face ? 1.0/mat.refractiveIndex : mat.refractiveIndex;
    vec3 unitDirection = unit_vector(r_in.GetDirection());
    float cosTheta = fminf(dot(-unitDirection, rec.normal), 1.0f);
    float sinTheta = sqrtf(1.0f - cosTheta*cosTheta);
    vec3 direction;
    if (ri * sinTheta > 1.0f || reflectance(cosTheta, ri) > getRandomFloat(idx,states)) {
        // reflect
        direction = reflect(unitDirection, rec.normal);
    }else {
        // refract
        direction = refract(unitDirection, rec.normal,ri);
    }
    scattered = ray(rec.p, direction,r_in.getTime());
    return true;
}

__device__ bool scatter(int idx, curandState *states,material mat,const ray& rayIn,const hitRecord& rec,color& attenuation,ray& rayOut) {
    if (mat.type == materialType::METAL) {
        return metalScatter(idx,states,mat.metal,rayIn,rec,attenuation,rayOut);
    }else if (mat.type == materialType::LAMBERTIAN) {
        return lambertianScatter(idx,states,mat.lambertian,rayIn,rec,attenuation,rayOut);
    }else {
        return dielectricScatter(idx,states,mat.dielectric,rayIn,rec,attenuation,rayOut);
    }
}
#endif //MATERIAL_CUH
