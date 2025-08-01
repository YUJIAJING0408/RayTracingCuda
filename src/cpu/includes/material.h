//
// Created by YUJIAJING on 25-7-30.
//

#ifndef MATERIAL_H
#define MATERIAL_H

class materialCpu {
public:
    virtual ~materialCpu() = default;

    virtual bool scatter(
        const rayCpu& r_in, const hitRecordCpu& rec, color& attenuation, rayCpu& scattered
    ) const {
        printf("virtual material\n");
        return false;
    }
};



class lambertianCpu : public materialCpu {
public:
    lambertianCpu(const color& albedo) : albedo(albedo) {}
    bool scatter(const rayCpu& r_in, const hitRecordCpu& rec, color& attenuation, rayCpu& scattered)
    const override {
        vec3cpu scatterDirection = rec.normal + random_unit_vector();
        if (scatterDirection.nearZero()) {
            scatterDirection = rec.normal;
        }
        scattered = rayCpu(rec.p, scatterDirection);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

class metalCpu : public materialCpu {
public:
    metalCpu(const color& albedo,const float fz) : albedo(albedo),fuzz(fz) {}

    bool scatter(const rayCpu& r_in, const hitRecordCpu& rec, color& attenuation, rayCpu& scattered)
    const override {
        vec3cpu reflected = reflect(r_in.GetDirection(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz*random_unit_vector());
        scattered = rayCpu(rec.p, reflected);
        attenuation = albedo;
        return dot(scattered.GetDirection(), rec.normal) > 0;
    }

private:
    color albedo;
    float fuzz;
};

class dielectric : public materialCpu {
public:
    dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const rayCpu& r_in, const hitRecordCpu& rec, color& attenuation, rayCpu& scattered)
    const override {
        attenuation = color(1.0, 1.0, 1.0);
        float ri = rec.front_face ? (1.0/refraction_index) : refraction_index;

        vec3cpu unit_direction = unit_vector(r_in.GetDirection());
        float cos_theta = std::fminf(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = std::sqrtf(1.0f- cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3cpu direction;

        if (cannot_refract||reflectance(cos_theta, ri) > randomFloat())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered = rayCpu(rec.p, direction);
        return true;
    }

private:
    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    float refraction_index;
    static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0*r0;
        return r0 + (1-r0)*std::pow((1 - cosine),5);
    }
};
#endif //MATERIAL_H
