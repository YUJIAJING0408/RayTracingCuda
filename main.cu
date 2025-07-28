#include <iostream>


#include "raytracingCommon.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


int main() {
    std::cout << "Hello, World!" << std::endl;

    // World
    hittableList world;
    world.add(make_shared<sphere>(point3(0.f,0.f,-1.f),0.5));
    world.add(make_shared<sphere>(point3(0.f,-100.5f,-1.f), 100.f));

    // Camera
    camera cam;
    cam.aspectRatio = 16.0/9.0;
    cam.imageWidth = 400;
    cam.render("image.ppm",world);

    return 0;
}
