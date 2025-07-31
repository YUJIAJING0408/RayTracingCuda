//
// Created by YUJIAJING on 25-7-28.
//

#ifndef HITTABLELIST_H
#define HITTABLELIST_H
#include <memory>
#include "hittable.h"
#include "camera.h"
#include "shpere.h"
#include "vector"
using std::make_shared;
using std::shared_ptr;

class hittableListCpu : public hittableCpu {
public:
    vector<shared_ptr<hittableCpu>> objects;
    hittableListCpu(){};
    hittableListCpu(shared_ptr<hittableCpu> object) { add(object); }

    void add(shared_ptr<hittableCpu> object) {
        objects.push_back(object);
    }

    bool hit(const rayCpu &r, intervalCpu rayT, hitRecordCpu &rec) const override {
        hitRecordCpu tempRec;
        auto closest_so_far = rayT.max;
        bool hitAnything = false;
        for (const auto& object : objects) {
            if (object->hit(r, intervalCpu(rayT.min,closest_so_far), tempRec)) {
                hitAnything = true;
                closest_so_far = tempRec.t;
                rec = tempRec;
            }
        }
        return hitAnything;
    }
};


#endif //HITTABLELIST_H
