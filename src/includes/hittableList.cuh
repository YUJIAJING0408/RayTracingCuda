//
// Created by YUJIAJING on 25-7-28.
//

#ifndef HITTABLELIST_CUH
#define HITTABLELIST_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include <memory>
#include "hittable.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cameraCuda.cuh"
#include "shpere.cuh"
#include <thrust/sequence.h>
#include <thrust/copy.h>
using std::make_shared;
using thrust::device_vector;
using thrust::host_vector;
using std::shared_ptr;

class hittableList : public hittable {
public:
    vector<shared_ptr<hittable>> objects;
    hittableList(){};
    hittableList(shared_ptr<hittable> object) { add(object); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    CUDA_CALLABLE device_vector<sphere> toSphserList() {
        device_vector<sphere> sl ;
        for (int i = 0; i < this->objects.size(); i++) {
            sl.push_back(static_cast<sphere&>(*objects[i]));
        }
        return sl;
    }

    CUDA_CALLABLE bool hit(const ray &r, interval rayT, hitRecord &rec) const override {
        hitRecord tempRec;
        auto closest_so_far = rayT.max;
        bool hitAnything = false;
        for (const auto& object : objects) {
            if (object->hit(r, interval(rayT.min,closest_so_far), tempRec)) {
                hitAnything = true;
                closest_so_far = tempRec.t;
                rec = tempRec;
            }
        }
        return hitAnything;
    }
};


#endif //HITTABLELIST_CUH
