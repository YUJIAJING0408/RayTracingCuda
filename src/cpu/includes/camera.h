//
// Created by YUJIAJING on 25-7-30.
//

#ifndef CAMERA_H
#define CAMERA_H
#include "raytracingCommon.h"
#include <material.h>



class cameraCpu {
public:
    float aspectRatio = 1.0f;
    int imageWidth = 32*16,samplePerPixel = 10;

    void render(const string name,const hittableCpu& world) {
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
                    rayCpu r = getRay(x, y);
                    pixelColor += rayColor(r,MAXDEPTH,world);
                }
                writeColor(file,pixelColor * pixelSampleScale);
            }
        }
        file.put('\n');
        file.close();
    }

    rayCpu getRay(int x,int y) const{
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
        return rayCpu(rayOrigin, rayDirection);
    }

    color rayColor(const rayCpu& r,int depth,const hittableCpu& world) {
        if (depth <= 0)
            return color(0,0,0);
        hitRecordCpu rec;
        if (world.hit(r, intervalCpu(1e-10, infinity), rec)) {
            rayCpu scattered;
            color attenuation;
            if (rec.mat->scatter(r,rec,attenuation,scattered)) {
                return attenuation * rayColor(scattered,depth-1, world);
            }
            return color(0,0,0);
        }
        vec3cpu unitDirection = unit_vector(r.GetDirection());
        auto a = 0.5*(unitDirection.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }

    vec3cpu getPixel00_Loc() {return pixel00_Loc;}
    vec3cpu getPixelDeltaU(){return pixelDeltaU;}
    vec3cpu getPixelDeltaV(){return pixelDeltaV;}
    point3cpu getCenter(){return center;}
    int getImageHeight(){return imageHeight;}
private:
    int imageHeight = 1;
    point3cpu center = point3cpu(0, 0, 0),pixel00_Loc;
    vec3cpu pixelDeltaU,pixelDeltaV;
    float pixelSampleScale = .1f,focalLength = 1.0f;

    void initialize() {
        imageHeight = static_cast<int>(static_cast<float>(imageWidth) / aspectRatio);
        imageHeight = imageHeight > 0 ? imageHeight : 1;

        focalLength = 1.0f;
        auto viewportHeight = 2.0f;
        auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth)/static_cast<float>(imageHeight));
        pixelSampleScale = 1.0f / static_cast<float>(samplePerPixel);
        auto viewportU = vec3cpu(viewportWidth, 0.f, 0.f);
        auto viewportV = vec3cpu(0, -viewportHeight, 0.f);

        pixelDeltaU = viewportU / static_cast<float>(imageWidth);
        pixelDeltaV = viewportV / static_cast<float>(imageHeight);

        auto viewportUpperLeft = center - vec3cpu(0, 0, focalLength) - viewportU/2 - viewportV/2;
        pixel00_Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

    }

    vec3cpu sampleSquare()const{
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3cpu(randomFloat() - 0.5f, randomFloat() - 0.5f, 0);
    }
};

#endif //CAMERA_H
