//
// Created by YUJIAJING on 25-7-28.
//

#ifndef CAMERACUDA_CUH
#define CAMERACUDA_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "raytracingCommon.h"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "shpere.cuh"
using thrust::device_vector;
using thrust::host_vector;

struct cameraInfo {
    float aspectRatio,focalLength,pixelSampleScale;
    int imageWidth,imageHeight,samplesPerPixel;
    point3 center,pixel00_Loc;
    vec3 pixelDeltaU,pixelDeltaV;
};

__device__ float getRandomFloat(int idx,curandState *states) {
    curandState localState = states[idx];
    float r = curand_uniform(&localState);
    states[idx] = localState;
    return r;
}

__device__ ray* getRayS(int idx,curandState *states,cameraInfo cam,int x,int y){
    ray rayList[MAXSPP];
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error at%s:%d -> %s\n" , __FILE__ , __LINE__,cudaGetErrorString(err)) ;
    // }

    for (int i = 0; i < cam.samplesPerPixel; i++) {
        float xR = (getRandomFloat(idx,states) - 0.5f) * 2.0f; //[-0.5f,0.5f]
        float yR = (getRandomFloat(idx,states) - 0.5f) * 2.0f;
        auto rayDirection =cam.pixel00_Loc + (xR + x) * cam.pixelDeltaU + (yR + y) * cam.pixelDeltaV - cam.center;
        rayList[i] = ray(cam.center, rayDirection);
        // rayList[i].GetDirection().print();
        if (x == 0&&y==0) {
            printf("offset x y = (%f,%f)\n",xR,yR);
        }
    }
    return rayList;
}

__device__ ray getRay(int idx,curandState *states,cameraInfo cam,int x,int y){
    float xR = (getRandomFloat(idx,states) - 0.5f); //[-0.5f,0.5f]
    float yR = (getRandomFloat(idx,states) - 0.5f);

    auto rayDirection =cam.pixel00_Loc + (xR + x) * cam.pixelDeltaU + (yR + y) * cam.pixelDeltaV - cam.center;
    // if (x == 0&&y==0) {
    //     // printf("offset x y = (%f,%f)\n",xR,yR);
    //     printf("RayDir: (%.6f, %.6f, %.6f)\n", rayDirection.x(), rayDirection.y(), rayDirection.z());
    // }
    return ray(cam.center, rayDirection);
}

__device__ bool hitSphere(sphere &s,ray &r,interval rayT, hitRecord &rec) {
    vec3 oc = s.getCenter() - r.GetOrigin();
    // r.GetDirection().print();

    float a = r.GetDirection().lengthSquared();
    float h = dot(r.GetDirection(), oc);
    float c = oc.lengthSquared() - s.getRadius()*s.getRadius();

    float discriminant = h*h - a*c;
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
    vec3 outward_normal = (rec.p - s.getCenter()) / s.getRadius();
    rec.set_face_normal(r, outward_normal);
    return true;
}

__device__ bool hit(sphere* data, int size,ray &r, interval rayT, hitRecord &rec){
    hitRecord tempRec;
    auto closest_so_far = rayT.max;
    bool hitAnything = false;
    for (int i = 0; i < size; i++) {
        if (hitSphere(data[i],r, interval(rayT.min,closest_so_far), tempRec)) {
            // printf("r is %f\n",data[i].getRadius());
            hitAnything = true;
            closest_so_far = tempRec.t;
            rec = tempRec;
        }
    }
    return hitAnything;
}

__device__ color rayColor(ray& r,sphere* data, int size) {
    hitRecord rec;
    if (hit(data,size,r, interval(0.f, infinity), rec)) {
        return 0.5 * (rec.normal + color(1.f,1.f,1.f));
    }
    //
    vec3 unitDirection = unit_vector(r.GetDirection());
    auto a = 0.5*(unitDirection.y() + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__global__ void setup_random_states(curandState *states,int imageWidth,int imageHeight, unsigned long seed) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = h*imageWidth + w;
    if (idx < imageWidth*imageHeight) {
        // 每个线程使用不同的seed序列
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void kernelSample(curandState *states,cameraInfo cam,color* output,sphere* data, int size) {
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int idx = h*cam.imageWidth + w;
    if (w<cam.imageWidth && h<cam.imageHeight) {
        // printf("x = %d, y = %d\n", w, h);
        color pixelColor = color(0.f,0.f,0.f);
        // ray* rayList = getRayS(idx,states,cam,w,h);
        // printf("spp:%d",cam.samplesPerPixel );
        // bool sb[128];
        for (int i = 0; i < cam.samplesPerPixel; i++) {
            ray r = getRay(idx,states,cam,w,h);
            pixelColor += rayColor(r,data,size);
            // pixelColor += rayColor(rayList[i],data,size);
        }
        // avg
        pixelColor *= cam.pixelSampleScale;
        // save to mem
        int idx = w + h * cam.imageWidth;
        output[idx] = pixelColor;
    }
}

CUDA_CALLABLE void render(cameraInfo cam,const string name,device_vector<sphere> vectorSphere) {
    printf("world size = %llu\n",vectorSphere.size());
    ofstream file(name, ios::out);
    if (!file.is_open()) {
        // cout << "Error: Failed to open file for writing!" << endl;
        return;
    }
    file.write("P3\n", 3);
    auto widthAndHeight = to_string(cam.imageWidth) + " " + to_string(cam.imageHeight) + "\n";
    file.write(widthAndHeight.c_str(), widthAndHeight.size());
    file.write("255\n",4);
    dim3 block(32,32);
    dim3 grid( ceilf(static_cast<float>(cam.imageWidth)/32.f),ceilf(static_cast<float>(cam.imageHeight)/32.f));
    sphere* d_ptr = thrust::raw_pointer_cast(vectorSphere.data());
    size_t size = sizeof(color) * cam.imageWidth * cam.imageHeight;
    color* outputHost = static_cast<color *>(malloc(size));
    color* outputDevice;
    cudaError_t err = cudaMalloc((color **) &outputDevice, size);
    // random offset
    if (err!=cudaSuccess) {
        printf("分配cuda内存错误:%s\n",cudaGetErrorName(err));
    }

    //
    curandState *d_states;
    cudaMalloc(&d_states, cam.imageWidth * cam.imageHeight * sizeof(curandState));

    setup_random_states<<<grid, block>>>(d_states, cam.imageWidth,cam.imageHeight, time(0));

    kernelSample<<<grid,block>>>(d_states,cam,outputDevice,d_ptr,vectorSphere.size());
    cudaDeviceSynchronize();
    cudaMemcpy(outputHost, outputDevice, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < cam.imageWidth * cam.imageHeight; i++) {
        color pixel = outputHost[i];
        writeColor(file,pixel);
    }
    file.put('\n');
    file.close();
}

#endif //CAMERACUDA_CUH


