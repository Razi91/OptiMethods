//
// Created by jkonieczny on 14.06.15.
//

#ifndef METODY_UTILS_H
#define METODY_UTILS_H

#include <utility>

//typedef std::pair<double, double> pt;

class pt {
public:
    double x, y;
    pt(const double x, const double y) {
        this->x = x;
        this->y = y;
    }

    pt & operator+=(const pt &p){
        x += p.x;
        y += p.y;
        return *this;
    }
//    pt  operator- (const pt &p){
//        return pt(x-p.x, y-p.y);
//    }
//
//
//    pt & operator=(const pt &p){
//        x = p.x;
//        y = p.y;
//        return * this;
//    }

    double len(const pt &p) {
        return (p.x-x)*(p.x-x) + (p.y-y)*(p.y-y);
    }
};

//pt  operator* (const double s, const pt &p){
//    return pt(s*p.x, s*p.y);
//}

#endif //METODY_UTILS_H
