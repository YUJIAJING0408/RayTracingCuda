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
    sphere() :center(0,0,1), radius(0.2f){}
    sphere(const point3& c,float r): center(c), radius(r) {}
    CUDA_CALLABLE bool hit(const ray& r, interval rayT, hitRecord& rec) const override {
        // r.GetDirection().print();
        // printf("hittest \n");
        // printf("hittest \n");

        vec3 oc = center - r.GetOrigin();
        // r.GetDirection().print();
        float a = r.GetDirection().lengthSquared();
        float h = dot(r.GetDirection(), oc);
        float c = oc.lengthSquared() - radius*radius;

        float discriminant = h*h - a*c;
        // printf("a,h,c,d = %f,%f,%f,%f\n", a,h,c,discriminant);
        //
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
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
    CUDA_CALLABLE point3 getCenter() const {return center;}
    CUDA_CALLABLE float getRadius() const {return radius;}
private:
    point3 center;
    float radius;
};


#endif //SHPERE_H
