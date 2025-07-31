//
// Created by YUJIAJING on 25-7-28.
//

#ifndef HITTABLE_CUH
#define HITTABLE_CUH

class materialCpu ;

class hitRecordCpu {
public:
    point3cpu p; // hit point
    vec3cpu normal; // hit point normal
    shared_ptr<materialCpu> mat;
    float t; // hit time
    bool front_face;
    hitRecordCpu():p(.0f,.0f,.0f),normal(.0f,.0f,.0f),t(infinity),front_face(false){};
    void set_face_normal(const rayCpu& r, const vec3cpu& outward_normal) {
        front_face = dot(r.GetDirection(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittableCpu{
    public:
    virtual ~hittableCpu() = default;
    virtual bool hit(const rayCpu& r, intervalCpu rayT, hitRecordCpu& rec) const = 0;
};

#endif //HITTABLE_CUH
