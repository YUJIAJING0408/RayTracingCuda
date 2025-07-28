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

class camera {
public:
    float aspectRatio = 1.0f;
    int imageWidth = 100;

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
            // std::clog << "\rScanlines remaining: " << (imageHeight - y) << ' ' << std::flush;
            string line;
            for (int x = 0; x < imageWidth; x++) {
                auto pixelCenter = pixel00_Loc + static_cast<float>(x) * pixelDeltaU + static_cast<float>(y) * pixelDeltaV;
                auto rayDirection = pixelCenter-center;
                auto r = ray(center, rayDirection);
                auto pixelColor = rayColor(r,world);
                writeColor(file,pixelColor);
            }
        }
        file.put('\n');
        file.close();

    }
private:
    int imageHeight = 1;
    point3 center = point3(0, 0, 0),pixel00_Loc;
    vec3 pixelDeltaU,pixelDeltaV;


    CUDA_CALLABLE void initialize() {
        imageHeight = static_cast<int>(imageWidth / aspectRatio);
        imageHeight = imageHeight > 0 ? imageHeight : 1;

        auto focal_length = 1.0f;
        auto viewportHeight = 2.0f;
        auto viewportWidth = viewportHeight * (static_cast<float>(imageWidth)/imageHeight);

        auto viewportU = vec3(viewportWidth, 0.f, 0.f);
        auto viewportV = vec3(0, -viewportHeight, 0.f);

        pixelDeltaU = viewportU / imageWidth;
        pixelDeltaV = viewportV / imageHeight;

        auto viewportUpperLeft = center - vec3(0, 0, focal_length) - viewportU/2 - viewportV/2;
        pixel00_Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

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
};


#endif //CAMERA_CUH
