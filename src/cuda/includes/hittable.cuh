//
// Created by YUJIAJING on 25-7-28.
//

#ifndef HITTABLE_CUH
#define HITTABLE_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "raytracingCommon.cuh"
#include <algorithm>

enum class shapeType { SPHERE, PLANE, TRIANGLE };

struct sphereData {
    ray center;
    float radius;
    material material;
};

struct planeData {
    float y;
    material material;
};

struct shape {
    shapeType type;
    sphereData sphere;
    planeData plane;
    AABB aabb;
};

__host__ shape &makeMovingSphere(point3 c, point3 toPoint, float radius, material mat) {
    vec3 r = vec3(radius, radius, radius);
    ray center = ray(c, toPoint - c);
    AABB box1(center.at(0) - r, center.at(0) + r);
    AABB box2(center.at(1) - r, center.at(1) + r);
    shape sphere = {
        .type = shapeType::SPHERE,
        .sphere{
            .center = center,
            .radius = radius,
            .material = mat
        },
        .aabb = AABB(box1, box2)
    };
    return sphere;
}

__host__ shape &makeStaticSphere(point3 center, float radius, material mat) {
    shape sphere = {
        .type = shapeType::SPHERE,
        .sphere{
            .center = ray(center, vec3(0.f, 0.f, 0.f)),
            .radius = radius,
            .material = mat
        },
        .aabb = AABB(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius))
    };
    return sphere;
}

__device__ bool hitSphere(const sphereData &s, const ray &r, const interval rayT, hitRecord &rec) {
    point3 currentCenter = s.center.at(r.getTime());
    vec3 oc = currentCenter - r.GetOrigin();
    // r.GetDirection().print();
    float a = r.GetDirection().lengthSquared();
    float h = dot(r.GetDirection(), oc);
    float c = oc.lengthSquared() - s.radius * s.radius;

    float discriminant = h * h - a * c;
    // printf("a,h,c,d = %f,%f,%f,%f\n", a,h,c,discriminant);
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
    rec.mat = s.material;
    vec3 outward_normal = (rec.p - currentCenter) / s.radius;
    rec.set_face_normal(r, outward_normal);
    return true;
}

__device__ bool hit(const shape s, const ray &r, const interval rayT, hitRecord &rec) {
    switch (s.type) {
        case shapeType::SPHERE: {
            return hitSphere(s.sphere, r, rayT, rec);
        }
        case shapeType::PLANE: {
            //
            return false;
        }
        default:
            printf("未知形状");
            return false;
    }
}

__device__ bool hitShapeList(shape *s, int worldSize, const ray &r, const interval rayT, hitRecord &rec) {
    // make sure s is on gpu mem
    hitRecord tempRec{
        point3(0, 0, 0),
        vec3(0, 0, 0),
        {
            materialType::LAMBERTIAN, {
                color(1.0f, 0.0f, 0.0f),
            }
        }
    };
    auto closest_so_far = rayT.max;
    bool hitAnything = false;
    for (int i = 0; i < worldSize; i++) {
        if (hit(s[i], r, interval(rayT.min, closest_so_far), tempRec)) {
            hitAnything = true;
            closest_so_far = tempRec.t;
            rec = tempRec;
        }
    }

    return hitAnything;
}

// InCuda
// class bvhNodeCuda {
//     bvhNodeCuda* left;
//     bvhNodeCuda* right;
//     AABB box;
//
//     __host__ bvhNodeCuda(vector<shape>& shapes,size_t start,size_t end) {
//         //
//         int axis = randomInt(0, 2);
//         auto comparator = (axis == 0) ? boxXCompare : (axis ==1) ? boxYCompare : boxZCompare;
//         size_t objectSpan = end - start;
//         if (objectSpan==1) {
//             // to cuda
//             shape* s;
//             cudaMalloc(static_cast<shape**>(&s), sizeof(shape));
//             cudaMemcpy(s, shapes.data(), sizeof(shape),cudaMemcpyHostToDevice);
//             left = right = s;
//         }else if (objectSpan==2) {
//             shape* s;
//             cudaMalloc(static_cast<shape**>(&s), 2*sizeof(shape));
//             cudaMemcpy(s, shapes.data(), sizeof(shape),cudaMemcpyHostToDevice);
//             left = &s[0];
//             right = &s[1];
//         }else {
//             // sort
//             std::sort(shapes.begin(), shapes.end(), comparator);
//             auto mid = start +objectSpan/2;
//             left = bvh
//         }
//     }
// };
//
// class bvhLeafCuda {
//     shape* left;
//     shape* right;
//     AABB box;
//     bvhLeafCuda(vector<shape>& shapes,size_t start,size_t end) {
//         size_t objectSpan = end - start;
//         if (objectSpan==1) {
//             // to cuda
//             shape* s;
//             cudaMalloc(static_cast<shape**>(&s), sizeof(shape));
//             cudaMemcpy(s, shapes.data(), sizeof(shape),cudaMemcpyHostToDevice);
//             left = right = s;
//         }else {
//             shape* s;
//             cudaMalloc(static_cast<shape**>(&s), 2*sizeof(shape));
//             cudaMemcpy(s, shapes.data(), sizeof(shape),cudaMemcpyHostToDevice);
//             left = &s[0];
//             right = &s[1];
//         }
//     }
// };

struct bvhInCuda {
    bvhInCuda *leftNode;
    bvhInCuda *rightNode;
    bool isLeaf;
    AABB aabb;
    shape* l;
    shape* r;
};

__device__ AABB getBvhAABB(bvhInCuda* bvh) {
    if (bvh->isLeaf) {
        return bvh->aabb;
    }
    return AABB(bvh->leftNode->aabb, bvh->rightNode->aabb);
}


__device__ bool hitBvh(bvhInCuda bvh,const ray& r, interval rayT, hitRecord& rec) {
    if (!getBvhAABB(&bvh).hit(r, rayT)) {
        return false;
    }
    bool hitLeft,hitRight;
    if (!bvh.isLeaf) {
        hitLeft = hitBvh(*bvh.leftNode,r,rayT,rec);
        hitRight = hitBvh(*bvh.rightNode,r,interval(rayT.min, hitLeft ? rec.t : rayT.max),rec);
        return hitLeft || hitRight;
    }
    // isLeaf
    shape leftShape = *bvh.l;
    shape rightShape = *bvh.r;
    hitLeft = hit(leftShape,r,rayT,rec);
    hitRight = hit(rightShape,r,interval(rayT.min, hitLeft ? rec.t : rayT.max),rec);
    return hitLeft || hitRight;
}

__host__ bool boxCompare(shape a,shape b,int axisIndex) {
    auto aAxisInterval = a.aabb.axisInterval(axisIndex);
    auto bAxisInterval = b.aabb.axisInterval(axisIndex);
    return aAxisInterval.min <bAxisInterval.min;
}

__host__ bool boxXCompare(shape a,shape b) {
    return boxCompare(a,b,0);
}

__host__ bool boxYCompare(shape a,shape b) {
    return boxCompare(a,b,1);
}

__host__ bool boxZCompare(shape a,shape b) {
    return boxCompare(a,b,2);
}

__host__ bvhInCuda makeBvhInCuda(vector<shape> &objects, size_t start, size_t end) {
    bvhInCuda b = {
    };
    int axis = randomInt(0, 2);

    auto comparator = (axis == 0)? boxXCompare: (axis == 1)? boxYCompare: boxZCompare;
    size_t object_span = end - start;
    if (object_span == 1) {
        b.isLeaf = true;
        shape *tmp;
        cudaMalloc(static_cast<shape **>(&tmp), sizeof(shape));
        cudaMemcpy(tmp, &objects[start], sizeof(shape), cudaMemcpyHostToDevice);
        b.leftNode = nullptr;
        b.rightNode = nullptr;
        b.l = b.r = tmp;
        b.aabb = objects[0].aabb;
    } else if (object_span == 2) {
        b.isLeaf = true;
        shape *tmp;
        cudaMalloc(static_cast<shape **>(&tmp), 2 * sizeof(shape));
        cudaMemcpy(tmp, &objects[start], 2 * sizeof(shape), cudaMemcpyHostToDevice);
        b.leftNode = nullptr;
        b.rightNode = nullptr;
        b.l = &tmp[0];
        b.r = &tmp[1];
        b.aabb = AABB(objects[0].aabb, objects[1].aabb);
    } else {
        std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);
        auto mid = start + object_span / 2;
        bvhInCuda* bvhLeft;
        bvhInCuda* bvhRight;
        cudaMalloc(static_cast<bvhInCuda**>(&bvhLeft), sizeof(bvhInCuda));
        cudaMalloc(static_cast<bvhInCuda**>(&bvhRight), sizeof(bvhInCuda));
        bvhInCuda bvhLeftHost = makeBvhInCuda(objects, start, mid);
        bvhInCuda bvhRightHost = makeBvhInCuda(objects, mid, end);
        cudaMemcpy(bvhLeft,&bvhLeftHost,sizeof(bvhInCuda),cudaMemcpyHostToDevice);
        cudaMemcpy(bvhRight,&bvhRightHost,sizeof(bvhInCuda),cudaMemcpyHostToDevice);
        b.leftNode = bvhLeft;
        b.rightNode = bvhRight;
        b.isLeaf = false;
        b.l = b.r = nullptr;
    }
    return b;
}




#endif //HITTABLE_CUH
