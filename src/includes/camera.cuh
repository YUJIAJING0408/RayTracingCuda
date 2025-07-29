//
// Created by YUJIAJING on 25-7-28.
//

#ifndef CAMERA_CUH
#define CAMERA_CUH
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include "raytracingCommon.h"
#include <thrust/device_vector.h>


using thrust::device_vector;

class camera {
public:
    float aspectRatio = 1.0f;
    int imageWidth = 32*16,samplePerPixel = 10;

    CUDA_CALLABLE cameraInfo toCameraInfo() {
        initialize();
        cameraInfo c;
        c.imageWidth = imageWidth;
        c.imageHeight = imageHeight;
        c.aspectRatio = aspectRatio;
        c.samplesPerPixel = samplePerPixel;
        c.center = center;
        c.focalLength = focalLength;
        c.pixelSampleScale = pixelSampleScale;
        c.pixel00_Loc = pixel00_Loc;
        c.pixelDeltaU = pixelDeltaU;
        c.pixelDeltaV = pixelDeltaV;
        return c;
    }

    CUDA_CALLABLE void render(const string name,const hittable& world) {
        initialize();
        ofstream file(name, ios::out);
        if (!file.is_open()) {
            // cout << "Error: Failed to open file for writing!" << endl;
            return;
        }
        file.write("P3\n", 3);
        auto widthAndHeight = to_string(imageWidth) + " " + to_string(imageHeight) + "\n";
        file.write(widthAndHeight.c_str(), widthAndHeight.size());
        file.write("255\n",4);
        for (int y = 0; y < imageHeight; y++) {
            progressBar(y,imageHeight,50);
            string line;
            for (int x = 0; x < imageWidth; x++) {
                color pixelColor(0, 0, 0);
                for (int sample = 0; sample < samplePerPixel; sample++) {
                    ray r = getRay(x, y);
                    pixelColor += rayColor(r,world);
                }
                writeColor(file,pixelColor * pixelSampleScale);
            }
        }
        file.put('\n');
        file.close();
    }

    CUDA_CALLABLE ray getRay(int x,int y) const{
        auto offset = sampleSquare();

        auto pixelSample = pixel00_Loc + (offset.x() + x) * pixelDeltaU + (offset.y() + y) * pixelDeltaV;
        auto rayOrigin = center;
        // rayOrigin.print();
        auto rayDirection = pixelSample - rayOrigin;
        // rayDirection.print();
        // if (x == 0&&y==0) {
        //     // printf("offset x y = (%f,%f)\n",offset.x(),offset.y());
        //     printf("RayDir: (%.6f, %.6f, %.6f)\n", rayDirection.x(), rayDirection.y(), rayDirection.z());
        // }
        return ray(rayOrigin, rayDirection);
    }

    CUDA_CALLABLE color rayColor(const ray& r,const hittable& world) {
        hitRecord rec;
        if (world.hit(r, interval(0, infinity), rec)) {
            return 0.5 * (rec.normal + color(1.f,1.f,1.f));
        }
        vec3 unitDirection = unit_vector(r.GetDirection());
        auto a = 0.5*(unitDirection.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }

    CUDA_CALLABLE vec3 getPixel00_Loc() {return pixel00_Loc;}
    CUDA_CALLABLE vec3 getPixelDeltaU(){return pixelDeltaU;}
    CUDA_CALLABLE vec3 getPixelDeltaV(){return pixelDeltaV;}
    CUDA_CALLABLE point3 getCenter(){return center;}
    CUDA_CALLABLE int getImageHeight(){return imageHeight;}
private:
    int imageHeight = 1;
    point3 center = point3(0, 0, 0),pixel00_Loc;
    vec3 pixelDeltaU,pixelDeltaV;
    float pixelSampleScale = .1f,focalLength = 1.0f;

    CUDA_CALLABLE void initialize() {
        imageHeight = static_cast<int>(imageWidth / aspectRatio);
        imageHeight = imageHeight > 0 ? imageHeight : 1;

        focalLength = 1.0f;
        auto viewportHeight = 2.0f;
        auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth)/imageHeight);
        pixelSampleScale = 1.0f / samplePerPixel;
        auto viewportU = vec3(viewportWidth, 0.f, 0.f);
        auto viewportV = vec3(0, -viewportHeight, 0.f);

        pixelDeltaU = viewportU / imageWidth;
        pixelDeltaV = viewportV / imageHeight;

        auto viewportUpperLeft = center - vec3(0, 0, focalLength) - viewportU/2 - viewportV/2;
        pixel00_Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

    }

    CUDA_CALLABLE vec3 sampleSquare()const{
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(randomFloat() - 0.5, randomFloat() - 0.5, 0);
    }
};
#endif //CAMERA_CUH


