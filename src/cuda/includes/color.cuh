//
// Created by YUJIAJING on 25-7-28.
//

#ifndef COLOR_CUH
#define COLOR_CUH
#pragma once
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
#include <fstream>
#include "interval.cuh"
#include "vec3.cuh"
#include <string>

using namespace std;

using color = vec3;

CUDA_CALLABLE inline double linear2Gamma(double linearComponent)
{
    return linearComponent > 0?sqrtf(linearComponent):0;
}

inline color colorRandom() {
    return color(randomFloat(),randomFloat(),randomFloat());
}

inline color colorRandom(float min,float max ) {
    return color(randomFloat(min,max),randomFloat(min,max),randomFloat(min,max));
}

CUDA_CALLABLE void writeColor(std::ofstream &file,const color& pixelColor) {
    string line;
    auto r = linear2Gamma( pixelColor.x());
    auto g = linear2Gamma(pixelColor.y());
    auto b = linear2Gamma(pixelColor.z());
    const interval intensity(0.000, 0.999);
    int ir = int(256 * intensity.clamp(r));
    int ig = int(256 * intensity.clamp(g));
    int ib = int(256 * intensity.clamp(b));
    line += to_string(ir) + " " + to_string(ig) + " " + to_string(ib) + " ";
    // cout<<line<<endl;
    file.write(line.c_str(), line.length());
}

#endif //COLOR_CUH
