#include <iostream>


#include "raytracingCommon.cuh"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif


int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, CUDALIMITSTACKSIZE);
    std::cout << "Hello, World!" << std::endl;
    shape sphereCenter = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(0.f, 0.f, -1.2f),
            .radius = 0.5f,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(0.1, 0.2, 0.5),
                }
            }
        }
    };
    shape ball = {
        .type = shapeType::SPHERE,
        .sphere{
            .center = point3(0.f, -100.5f, -1.f),
            .radius = 100.f,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(0.8, 0.8, 0.0)
                },
                .metal = {
                    .albedo = color(0.8, 0.8, 0.8),
                }
            }
        }

    };
    shape sphereLeft = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(-1.0, 0.0, -1.0),
            .radius = 0.5f,
            .material = {
                .type = materialType::DIELECTRIC,
                .metal = {
                    .albedo = color(0.8, 0.8, 0.8),
                    .fuzz = 0.3
                },
                .dielectric = {
                  .refractiveIndex=1.5f
                }
            }
        }
    };
    shape sphereLeftBubble = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(-1.0, 0.0, -1.0),
            .radius = 0.4f,
            .material = {
                .type = materialType::DIELECTRIC,
                .metal = {
                    .albedo = color(0.8, 0.8, 0.8),
                    .fuzz = 0.3
                },
                .dielectric = {
                    .refractiveIndex=1.0f/1.5f
                  }
            }
        }
    };
    shape sphereRight = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(1.0, 0.0, -1.0),
            .radius = 0.5f,
            .material = {
                .type = materialType::METAL,
                .metal = {
                    .albedo = color(0.8, 0.6, 0.2),
                    .fuzz = 1.0
                }
            }
        }
    };

    // new host mem
    vector<shape> hostWorld;
    hostWorld.push_back(sphereCenter);
    hostWorld.push_back(ball);
    hostWorld.push_back(sphereLeft);
    hostWorld.push_back(sphereLeftBubble);
    hostWorld.push_back(sphereRight);

    // to device mem
    shape *deviceWorld = nullptr;
    // malloc gpu mem
    cudaError_t err = cudaMalloc(reinterpret_cast<shape **>(&deviceWorld), hostWorld.size() * sizeof(shape));
    if (isCudaError(err,__FILE__,__LINE__)) {
        return -1;
    }
    // cpy from host to device
    err = cudaMemcpy(deviceWorld,hostWorld.data(), hostWorld.size() *sizeof(shape), cudaMemcpyHostToDevice);
    if(isCudaError(err,__FILE__,__LINE__)) {
        return -1;
    }
    // hostWorld.shrink_to_fit(); // free host mem
    if(isCudaError(cudaGetLastError(),__FILE__,__LINE__)) {
        return -1;
    }
    cameraCuda cam = newCamera(IMAGEWIDTH,MAXSPP, 2.0 / 1.0);
    printf("Camera pss:%f\n",cam.pixelSampleScale);
    render(cam, "imageCuda.ppm", deviceWorld, static_cast<int>(hostWorld.size()));
    return 0;
}