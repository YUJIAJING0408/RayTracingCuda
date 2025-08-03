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
#include "raytracingCommon.cuh"
// #include "raytracingCommon.cuh"
#include <curand_kernel.h>

struct cameraCuda {
    float aspectRatio,pixelSampleScale,vfov,defocusAngle = 0.f,focusDist = 10.f;
    int imageWidth,imageHeight,samplesPerPixel;
    point3 center,pixel00_Loc,lookFrom,lookAt = point3(.0f,.0f,-1.f);
    vec3 pixelDeltaU,pixelDeltaV,u,v,w,vup = vec3(0.0f,1.0f,0.0f),defocusDiskU,defocusDiskV;
};

cameraCuda newCamera(int imageWidth,int samplesPerPixel,float aspectRatio,float vfov,float defocusAngle,float focusDisk,point3 lookFrom,point3 lookAt,vec3 vup) {
    cameraCuda c;
    c.vfov=vfov;
    c.vup = vup;
    c.imageWidth = imageWidth;
    c.aspectRatio = aspectRatio;
    int imageHeight = static_cast<int>(static_cast<float>(imageWidth) / aspectRatio);
    c.imageHeight = imageHeight > 0 ? imageHeight : 1;
    c.samplesPerPixel = samplesPerPixel;
    c.center = lookFrom;
    c.lookFrom = lookFrom;
    c.defocusAngle = defocusAngle;
    c.focusDist = focusDisk;
    float theta = degrees2Radians(vfov);
    float h = tanf(theta/2.f);

    auto viewportHeight = 2.0f * h * c.focusDist;
    auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth)/static_cast<float>(imageHeight));
    c.pixelSampleScale = 1.0f / static_cast<float>(samplesPerPixel);
    c.w = unit_vector(lookFrom-lookAt);
    c.u = unit_vector(cross(c.vup,c.w));
    c.v = cross(c.w,c.u);
    auto viewportU = viewportWidth * c.u;
    auto viewportV = -viewportHeight * c.v;
    c.pixelDeltaU = viewportU / static_cast<float>(imageWidth);
    c.pixelDeltaV = viewportV / static_cast<float>(imageHeight);

    auto viewportUpperLeft = c.center - c.focusDist * c.w - viewportU/2 - viewportV/2;
    c.pixel00_Loc = viewportUpperLeft + 0.5 * (c.pixelDeltaU + c.pixelDeltaV);

    float defocusRadius = focusDisk * tanf(degrees2Radians(defocusAngle/2.f));
    c.defocusDiskU = c.u * defocusRadius;
    c.defocusDiskV = c.v * defocusRadius;
    return c;
};

__device__ point3 defocusDiskSample(int idx,curandState *states,cameraCuda cam) {
    point3 p = randomInUnitDisk(idx,states);
    return cam.center + (p.x() * cam.defocusDiskU) + (p.y() * cam.defocusDiskV);
}

__device__ ray getRay(int idx,curandState *states,cameraCuda cam,int x,int y){
    float xR = (getRandomFloat(idx,states) - 0.5f); //[-0.5f,0.5f]
    float yR = (getRandomFloat(idx,states) - 0.5f);

    auto rayDirection =cam.pixel00_Loc + (xR + x) * cam.pixelDeltaU + (yR + y) * cam.pixelDeltaV - cam.center;
    point3 origin= (cam.defocusAngle <=0)?cam.center:defocusDiskSample(idx,states,cam);
    float t = getRandomFloat(idx,states);
    return ray(origin, rayDirection,t);
}

__device__ color rayColor(int idx,curandState *states,ray r,int depth,shape* world,int worldSize,bvhInCuda * bvh) {
    // printf("depth = %d\n",depth);
    if (depth <= 0)
        return color(0,0,0);
    hitRecord rec = {
        point3(0, 0, 0),
        vec3(0, 0, 0),
        {
            materialType::LAMBERTIAN, {
                 .albedoTexture=makeSolidTexture(color(1.0f, 0.0f, 0.0f)) ,
            }
        }
    };
    if (bvh!=nullptr) {
        // bvh
        if (hitBvh(*bvh,r,interval(1e-3f, infinity), rec)) {
            ray scattered;
            color attenuation;
            // printf("material m %d\n",m.type);

            if (scatter(idx,states,rec.mat,r,rec,attenuation,scattered)) {
                return attenuation * rayColor(idx,states,scattered,depth-1, world,worldSize,bvh);
            }
            return color(0,0,0);
        }
    }else {
        if (hitShapeList(world,worldSize,r, interval(1e-3f, infinity), rec)) {
            // return 0.5 * (rec.normal + color(1.f,1.f,1.f));
            // vec3 direction = getRandomOnHemisphere(idx,states,rec.normal);
            // vec3 direction  = rec.normal + getRandomUnitVec3(idx,states);
            // ray r = ray(rec.p,direction);
            // return 0.5 * rayColor(idx,states,r,depth-1,world,worldSize);
            ray scattered;
            color attenuation;
            // printf("material m %d\n",m.type);

            if (scatter(idx,states,rec.mat,r,rec,attenuation,scattered)) {
                return attenuation * rayColor(idx,states,scattered,depth-1, world,worldSize,nullptr);
            }

            return color(0,0,0);
        }
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

__global__ __launch_bounds__(1024,4) void kernelSample(curandState *states,cameraCuda cam,color * devicePixels,shape* world,int worldSize,bvhInCuda* bvh) {
    // printf("world size = %d\n",worldSize);
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int w = threadIdx.x + blockIdx.x * blockDim.x;
    const int idx = h*cam.imageWidth + w;
    if (idx == 0) {
        printf("idx = 0\n");
        // world[0].data.sphere.material.lambertian.albedo.print();
        // world[1].data.sphere.material.lambertian.albedo.print();
        // world[2].data.sphere.material.metal.albedo.print();
        // world[3].data.sphere.material.metal.albedo.print();
    }
    // printf("w:%d,h:%d,idx:%d\n",w,h,idx);
    if (w<cam.imageWidth && h<cam.imageHeight) {
        color pixelColor = color(0.f,0.f,0.f);
        for (int i = 0; i < cam.samplesPerPixel; i++) {
            ray r = getRay(idx,states,cam,w,h);
            pixelColor += rayColor(idx,states,r,MAXDEPTH,world,worldSize,bvh);
        }
        // avg
        pixelColor *= cam.pixelSampleScale;
        // save to device mem
        devicePixels[idx] = pixelColor;
    }
}

CUDA_CALLABLE void render(cameraCuda cam,const string name,shape* deviceWorld,int worldSize,bvhInCuda* bvh) {
    printf("world size = %d\n",worldSize);
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
    dim3 grid( ceilf(static_cast<float>(cam.imageWidth)/block.x),ceilf(static_cast<float>(cam.imageHeight)/block.y));

    // to device
    size_t pixelSize = cam.imageWidth * cam.imageHeight;
    color * devicePixels;
    color * hostPixels;
    hostPixels = static_cast<color *>(malloc(pixelSize * sizeof(color)));
    cudaError_t err =cudaMalloc(static_cast<color**>(&devicePixels), pixelSize*sizeof(color));
    if (isCudaError(err,__FILE__,__LINE__)) {
        return ;
    }
    // random
    curandState *d_states;
    err = cudaMalloc(&d_states, cam.imageWidth * cam.imageHeight * sizeof(curandState));
    if (isCudaError(err,__FILE__,__LINE__)) {
        return;
    }
    setup_random_states<<<grid, block>>>(d_states, cam.imageWidth,cam.imageHeight, time(0));
    // render in gpu
    if (isCudaError(cudaGetLastError(),__FILE__,__LINE__)) {
        return;
    }
    kernelSample<<<grid,block>>>(d_states,cam,devicePixels,deviceWorld,worldSize,bvh);
    cudaDeviceSynchronize();
    if (isCudaError(cudaGetLastError(),__FILE__,__LINE__)) {
        return;
    }
    if (isCudaError(cudaMemcpy(hostPixels, devicePixels, pixelSize*sizeof(color), cudaMemcpyDeviceToHost)
    ,__FILE__,__LINE__)) {
        return;
    }
    for (int i = 0; i < cam.imageWidth*cam.imageHeight; i++) {
        color pixel = hostPixels[i];
        writeColor(file,pixel);
    }
    // device free
    cudaFree(devicePixels);
    cudaFree(deviceWorld);
    cudaFree(d_states);
    // host free
    free(hostPixels);
    file.put('\n');
    file.close();
}

#endif //CAMERACUDA_CUH


