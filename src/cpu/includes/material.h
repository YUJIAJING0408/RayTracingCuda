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
    metalCpu(const color& albedo) : albedo(albedo) {}

    bool scatter(const rayCpu& r_in, const hitRecordCpu& rec, color& attenuation, rayCpu& scattered)
    const override {
        vec3cpu reflected = reflect(r_in.GetDirection(), rec.normal);
        scattered = rayCpu(rec.p, reflected);
        attenuation = albedo;
        return true;
    }

private:
    color albedo;
};

#endif //MATERIAL_H
