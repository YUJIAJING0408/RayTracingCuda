#include <iostream>

#include "color.cuh"
#include "ray.cuh"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

CUDA_CALLABLE color rayColor(const ray& r) {
    vec3 unitDirect = unit_vector(r.GetDirection());
    auto a = 0.5*(unitDirect.y() + 1.0);
    return (1.0-a)*color(1.f, 1.f, 1.f) + a*color(0.5f, 0.7f, 1.0f);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    auto aspectRatio = 16.0f / 9.0f;
    int imageWidth = 400;
    int imageHeight = static_cast<int>(static_cast<float>(imageWidth) / aspectRatio);
    imageHeight = imageHeight>1 ? imageHeight : 1;

    // Camera
    auto focalLength = 2.0f;
    auto viewportHeight = 2.0f;
    auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth) / static_cast<float>(imageHeight));
    auto cameraCenter = point3(0,0,0);

    auto viewportU = vec3(viewportWidth,0,0);
    auto viewportV = vec3(0,-viewportHeight,0);

    auto pixelDeltaU = viewportU / static_cast<float>(imageWidth);
    auto pixelDeltaV = viewportV / static_cast<float>(imageHeight);
    //
    auto viewportUpperLeft = cameraCenter - vec3(.0f,.0f,focalLength) - viewportU / 2.0f - viewportV / 2.0f ;
    auto pixel00_Loc = viewportUpperLeft + (pixelDeltaU + pixelDeltaV) * 0.5f;

    const string fileName = "image.ppm";
    ofstream file(fileName, ios::out);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file for writing!" << std::endl;
        return 1;
    }
    file.write("P3\n", 3);
    auto widthAndHeight = to_string(imageWidth) + " " + to_string(imageHeight) + "\n";
    file.write(widthAndHeight.c_str(), widthAndHeight.size());
    file.write("255\n",4);
    for (int y = 0; y < imageHeight; y++) {
        std::clog << "\rScanlines remaining: " << (imageHeight - y) << ' ' << std::flush;
        string line;
        for (int x = 0; x < imageWidth; x++) {
            auto pixelCenter = pixel00_Loc + static_cast<float>(x) * pixelDeltaU + static_cast<float>(y) * pixelDeltaV;
            auto rayDirection = pixelCenter-cameraCenter;
            auto r = ray(cameraCenter, rayDirection);
            auto pixelColor = rayColor(r);
            writeColor(file,pixelColor);
        }
    }
    file.put('\n');
    file.close();
    return 0;
}
