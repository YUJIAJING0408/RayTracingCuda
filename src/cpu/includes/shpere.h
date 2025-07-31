//
// Created by YUJIAJING on 25-7-28.
//

#ifndef SHPERE_H
#define SHPERE_H

#include <utility>
#include "material.h"
class sphereCpu : public hittableCpu {
public:
    sphereCpu() :center(0,0,1), radius(0.2f){}
    sphereCpu(const point3cpu& c,float r,shared_ptr<materialCpu> m): center(c), radius(r),_material(std::move(m)) {}
    bool hit(const rayCpu& r, intervalCpu rayT, hitRecordCpu& rec) const override {
        // r.GetDirection().print();
        // printf("hittest \n");
        // printf("hittest \n");

        vec3cpu oc = center - r.GetOrigin();
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
        rec.mat=_material;
        vec3cpu outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        return true;
    }
    point3cpu getCenter() const {return center;}
    float getRadius() const {return radius;}
private:
    point3cpu center;
    float radius;
    shared_ptr<materialCpu> _material;
};


#endif //SHPERE_H
