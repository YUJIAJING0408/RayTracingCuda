//
// Created by YUJIAJING on 25-7-28.
//

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(){
    int width = 256, height = 256;
    const string fileName = "image.ppm";
    ofstream file(fileName, ios::out);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file for writing!" << std::endl;
        return 1;
    }
    file.write("P3\n", 3);
    auto widthAndHeight = to_string(width) + " " + to_string(height) + "\n";
    file.write(widthAndHeight.c_str(), widthAndHeight.size());
    file.write("255\n",4);
    for (int y = 0; y < height; y++) {
        std::clog << "\rScanlines remaining: " << (height - y) << ' ' << std::flush;
        string line;
        for (int x = 0; x < width; x++) {
            auto r = double(x) / (width-1);
            auto g = double(y) / (height-1);
            auto b = 0.0;
            int ir = int(255.999 * r);
            int ig = int(255.999 * g);
            int ib = int(255.999 * b);
            line += to_string(ir) + " " + to_string(ig) + " " + to_string(ib) + " ";
        }
        file.write(line.c_str(), line.length());
    }
    // 可选：添加换行符使文件符合标准文本格式
    file.put('\n');
    file.close();
}