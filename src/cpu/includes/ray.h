//
// Created by YUJIAJING on 25-7-28.
//

#ifndef RAY_H
#define RAY_H

class rayCpu {
public:
    rayCpu(): origin(0,0,0), direction(0,0,-1) {}
    rayCpu(vec3cpu origin, vec3cpu direction): origin(origin), direction(direction) {}
    const point3cpu& GetOrigin() const { return origin; }
    const vec3cpu& GetDirection() const { return direction; }
    point3cpu at(float t) const { return origin + t * direction; }

private:
    point3cpu origin;
    vec3cpu direction;
};

#endif //RAY_H
