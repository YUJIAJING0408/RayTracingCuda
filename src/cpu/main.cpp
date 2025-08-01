//
// Created by YUJIAJING on 25-7-30.
//
#include "raytracingCommon.h"
#include <iostream>

using namespace std;

int main() {
    cout << "begin render at cpu!!!!!" << endl;
    // World
    hittableListCpu world;
    auto lg = make_shared<lambertianCpu> (color(0.8, 0.8, 0.0));
    auto lc = make_shared<lambertianCpu>(color(0.1, 0.2, 0.5));
    auto ml = make_shared<dielectric>(1.5f);
    auto mb = make_shared<dielectric>(1.00f / 1.50f);
    auto mr = make_shared<metalCpu>(color(0.8, 0.6, 0.2),1.0);
    world.add(make_shared<sphereCpu>(point3cpu(0.f, 0.f, -1.2f), 0.5f, lc));
    world.add(make_shared<sphereCpu>(point3cpu(0.f, -100.5f, -1.f), 100.f, lg));
    world.add(make_shared<sphereCpu>(point3cpu(-1.0, 0.0, -1.0), 0.5f, ml));
    world.add(make_shared<sphereCpu>(point3cpu(-1.0, 0.0, -1.0), 0.4f, mb));
    world.add(make_shared<sphereCpu>(point3cpu(1.0, 0.0, -1.0), 0.5f, mr));

    // Camera
    cameraCpu cam;
    // cam.aspectRatio = 16.0/9.0;
    cam.aspectRatio = 2.0 / 1.0;
    cam.imageWidth = IMAGEWIDTH;
    cam.samplePerPixel = MAXSPP;
    cam.render("imageCPU.ppm", world);
    printf("------------------------");
    return 0;
}
