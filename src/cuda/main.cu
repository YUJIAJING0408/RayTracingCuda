#include <iostream>


#include "raytracingCommon.cuh"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


void makeWorld(const int size, vector<shape> &world) {
    for (int i = -size; i < size; i++) {
        for (int j = -size; j < size; j++) {
            point3 center(i + 0.8f * randomFloat(), 0.2f,j + 0.8f * randomFloat());
            float r = randomFloat();
            if ((center - point3(4, 0.2f, 0)).length() > 0.9) {
                if (r < 0.8f) {
                    // lambertian
                    shape lambertian = makeMovingSphere(center,vec3(0.f, randomFloat(0.f,.5f),0.f),0.2f,makeLambertian(randomFloat() * colorRandom(0.f, 1.0f)));
                    world.push_back(lambertian);
                } else if (r < 0.95f) {
                    // metal
                    shape metal = makeStaticSphere(center,0.2f,makeMetal(colorRandom(0.5, 1.0),randomFloat()));
                    world.push_back(metal);
                } else {
                    // glass
                    shape glass = makeStaticSphere(center,0.2f,makeDielectric(1.5f));
                    world.push_back(glass);
                }
            }
        }
    }
}


int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, CUDALIMITSTACKSIZE);
    std::cout << "Hello, World!" << std::endl;
    vector<shape> hostWorld;
    makeWorld(11, hostWorld);
    shape metalBall = makeStaticSphere(point3(4.f, 1.f, 0.f),1.f,makeMetal(color(0.7f, 0.6f, 0.5f),0.f));
    shape glassBall = makeStaticSphere(point3(0.f, 1.f, 0.f),1.f,makeDielectric(1.5f));
    shape lambertianBall = makeStaticSphere(point3(-4.f,1.f,0.f),1.0f,makeLambertian(color(0.4f,0.2f,0.1f)));
    shape groundBall = makeStaticSphere(point3(0.f, -1000.f, 0.f),1000.f,makeLambertian(color(0.5f, 0.5f, 0.5f)));


    hostWorld.push_back(groundBall);
    hostWorld.push_back(glassBall);
    hostWorld.push_back(metalBall);
    hostWorld.push_back(lambertianBall);

    bvhInCuda bvhHost = makeBvhInCuda(hostWorld,0,hostWorld.size()-1);
    bvhInCuda* bvhDevice;
    cudaMalloc(static_cast<bvhInCuda**>(&bvhDevice), sizeof(bvhInCuda));
    cudaMemcpy(bvhDevice,&bvhHost,sizeof(bvhInCuda),cudaMemcpyHostToDevice);
    // to device mem
    shape *deviceWorld = nullptr;
    // malloc gpu mem
    cudaError_t err = cudaMalloc(reinterpret_cast<shape **>(&deviceWorld), hostWorld.size() * sizeof(shape));
    if (isCudaError(err,__FILE__,__LINE__)) {
        return -1;
    }
    // cpy from host to device
    err = cudaMemcpy(deviceWorld, hostWorld.data(), hostWorld.size() * sizeof(shape), cudaMemcpyHostToDevice);
    if (isCudaError(err,__FILE__,__LINE__)) {
        return -1;
    }
    // hostWorld.shrink_to_fit(); // free host mem
    if (isCudaError(cudaGetLastError(),__FILE__,__LINE__)) {
        return -1;
    }
    cameraCuda cam = newCamera(IMAGEWIDTH,MAXSPP, 2.0 / 1.0, 20.f, .1f, 10.f, point3(13.f, 2.f, 3.f),
                               point3(0.0f, 0.0f, 0.0f), vec3(0.f, 1.0f, 0.f));
    printf("Camera pss:%f\n", cam.pixelSampleScale);

    render(cam, "imageCuda.ppm", deviceWorld, static_cast<int>(hostWorld.size()),nullptr);
    return 0;
}
