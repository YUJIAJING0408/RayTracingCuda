#include <iostream>

#include "color.cuh"
#include "ray.cuh"
#include "vec3.cuh"

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

color rayColor(const ray& r) {
    return color(0, 0, 0);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    auto aspectRatio = 16.0f / 9.0f;
    int imageWidth = 400;
    int imageHeight = static_cast<int>(static_cast<float>(imageWidth) / aspectRatio);
    imageHeight = imageHeight>1 ? imageHeight : 1; // 最小为1

    // Camera
    auto focalLength = 2.0f;
    auto viewportHeight = 2.0f;
    auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth) / static_cast<float>(imageHeight));
    auto cameraCenter = point3(0,0,0);

    // 视口长度
    auto viewportU = vec3(viewportWidth,0,0);
    auto viewportV = vec3(0,viewportHeight,0);

    // 像素间距
    auto pixelDeltaU = viewportU / static_cast<float>(imageWidth);
    auto pixelDeltaV = viewportV / static_cast<float>(imageHeight);

    // 视口左上角位置
    vec3 viewportUpperLeft;
    viewportUpperLeft = cameraCenter - vec3(0.0f,0.0f,focalLength) - viewportU / 2.0f - viewportV / 2.0f ;
    // 视口左上角第一个像素的中心
    vec3 pixel00_Loc = viewportUpperLeft + (pixelDeltaU + pixelDeltaV) * 0.5f;
    //
    // const string fileName = "image.ppm";
    // ofstream file(fileName, ios::out);
    // if (!file.is_open()) {
    //     std::cerr << "Error: Failed to open file for writing!" << std::endl;
    //     return 1;
    // }
    // file.write("P3\n", 3);
    // auto widthAndHeight = to_string(imageWidth) + " " + to_string(imageHeight) + "\n";
    // file.write(widthAndHeight.c_str(), widthAndHeight.size());
    // file.write("255\n",4);
    // // 像素点
    // for (int y = 0; y < imageHeight; y++) {
    //     std::clog << "\rScanlines remaining: " << (imageHeight - y) << ' ' << std::flush;
    //     string line;
    //     for (int x = 0; x < imageWidth; x++) {
    //         // 每次像素点位置通过偏移获得
    //         auto pixelCenter = pixel00_Loc + static_cast<float>(x) * pixelDeltaU + static_cast<float>(y) * pixelDeltaV;
    //         auto rayDirection = pixelCenter-cameraCenter;
    //         auto r = ray(cameraCenter, rayDirection);
    //         auto pixelColor = rayColor(r);
    //         writeColor(file,pixelColor);
    //     }
    // }
    // // 可选：添加换行符使文件符合标准文本格式
    // file.put('\n');
    // file.close();
    // return 0;
}
