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
                    shape lambertian = {
                        .type = shapeType::SPHERE,
                        .sphere = {
                            .center = ray(center,vec3(0.f, randomFloat(0.f,.5f), 0.f)),
                            .radius = 0.2f,
                            .material = {
                                .type = materialType::LAMBERTIAN,
                                .lambertian = {
                                    .albedo = randomFloat() * colorRandom(0.f, 1.0f)
                                }
                            }
                        }
                    };
                    world.push_back(lambertian);
                } else if (r < 0.95f) {
                    // metal
                    shape metal = {
                        .type = shapeType::SPHERE,
                        .sphere = {
                            .center = ray(center,vec3(0.f, 0.f, 0.f)),
                            .radius = 0.2f,
                            .material = {
                                .type = materialType::METAL,
                                .metal = {
                                    .albedo = colorRandom(0.5, 1.0),
                                    .fuzz = randomFloat()
                                },
                            }
                        }
                    };
                    world.push_back(metal);
                } else {
                    // glass
                    shape glass = {
                        .type = shapeType::SPHERE,
                        .sphere = {
                            .center = ray(center,vec3(0.f, 0.f, 0.f)),
                            .radius = 0.2f,
                            .material = {
                                .type = materialType::DIELECTRIC,
                                .dielectric = {
                                    .refractiveIndex = 1.5f
                                }
                            }
                        }
                    };
                    world.push_back(glass);
                }
            }
        }
    }
}


int main() {
    cudaDeviceSetLimit(cudaLimitStackSize, CUDALIMITSTACKSIZE);
    std::cout << "Hello, World!" << std::endl;

    /*
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
                    .fuzz = 0.3f
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
    */

    /*
    float R = cosf(pi/4.0f);
    shape sphereLeft = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(-R, 0.0f, -1.0f),
            .radius = R,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(0.f, 0.f, 1.f),
                }
            }
        }
    };
    shape sphereRight = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = point3(R, 0.0, -1.0),
            .radius = R,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(1.f, 0.f, 0.f)
                }
            }
        }
    };

    // new host mem
    vector<shape> hostWorld;
    hostWorld.push_back(sphereLeft);
    hostWorld.push_back(sphereRight);
    */

    vector<shape> hostWorld;
    makeWorld(11, hostWorld);
    shape metal = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = ray(point3(4.f, 1.f, 0.f),vec3(0.f,0.f,0.f)),
            .radius = 1.0f,
            .material = {
                .type = materialType::METAL,
                .metal = {
                    .albedo = color(0.7f, 0.6f, 0.5f),
                    .fuzz = 0.0f
                }
            }
        }
    };
    shape glass = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = ray(point3(0.f, 1.f, 0.f),vec3(0.f,0.f,0.f)),
            .radius = 1.0f,
            .material = {
                .type = materialType::DIELECTRIC,
                .dielectric = {
                    .refractiveIndex = 1.5f
                }
            }
        }
    };

    // lambertian
    shape lambertian = {
        .type = shapeType::SPHERE,
        .sphere = {
            .center = ray( point3(-4.f,1.f,0.f),vec3(0.f,0.f,0.f)),
            .radius = 1.0f,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(0.4f,0.2f,0.1f)
                }
            }
        }
    };

    shape groundBall = {
        .type = shapeType::SPHERE,
        .sphere{
            .center = ray(point3(0.f, -1000.f, 0.f),vec3(0.f,0.f,0.f)),
            .radius = 1000.f,
            .material = {
                .type = materialType::LAMBERTIAN,
                .lambertian = {
                    .albedo = color(0.5f, 0.5f, 0.5f)
                },
            }
        }

    };

    hostWorld.push_back(groundBall);
    hostWorld.push_back(glass);
    hostWorld.push_back(metal);
    hostWorld.push_back(lambertian);

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
    render(cam, "imageCuda.ppm", deviceWorld, static_cast<int>(hostWorld.size()));
    return 0;
}
