//
// Created by YUJIAJING on 25-7-28.
//

#ifndef SHPERE_H
#define SHPERE_H

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

class sphere : public hittable {
public:
    sphere(const point3& c,float r): center(c), radius(r) {}
    CUDA_CALLABLE bool hit(const ray& r, interval rayT, hitRecord& rec) const override {
        vec3 oc = center - r.GetOrigin();
        auto a = r.GetDirection().lengthSquared();
        auto h = dot(r.GetDirection(), oc);
        auto c = oc.lengthSquared() - radius*radius;

        auto discriminant = h*h - a*c;
        if (discriminant < 0)
            return false;
        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!rayT.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!rayT.surrounds(root)) {}
                return false;
        }
        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
private:
    point3 center;
    float radius;
};


#endif //SHPERE_H
