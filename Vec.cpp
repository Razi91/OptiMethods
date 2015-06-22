//
// Created by jkonieczny on 15.06.15.
//

#include "Vec.h"

#include <cassert>
#include <stdio.h>
#include <cmath>

Vec::Vec(int size) {
    data = new double[size];
    this->size = size;
}

Vec::~Vec() {
    delete data;
}

const double Vec::get(int i) {
    assert(i<size);
    return data[i];
}
const double Vec::operator()(const int i) {
    return get(i);
}

void Vec::set(int i, double v) {
    assert(i<size);
    data[i] = v;
}

const void Vec::dump() {
    printf("Vector(%d)\n [", size);
    for(int i=0; i<size; i++){
        printf("%lf ", get(i));
    }
    printf("]\n");
}

const int Vec::getSize() {
    return size;
}


Vec Vec::operator+(const Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)+b(i));
    }
    return v;
}

Vec Vec::operator-(const Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)-b(i));
    }
    return v;
}

Vec Vec::operator*(const Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)*b(i));
    }
    return v;
}

Vec Vec::operator/(const Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)/b(i));
    }
    return v;
}

Vec Vec::operator*(double b) {
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)*b);
    }
    return v;
}

Vec operator*(double a, const Vec &b) {
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, a*b(i));
    }
    return v;
}

Vec Vec::operator/(double b) {
    Vec v(size);
    for(int i=0; i<size; i++){
        v.set(i, get(i)/b);
    }
    return v;
}

double Vec::d() {
    double v = 0;
    for(int i=0; i<size; i++){
        v += get(i)*get(i);
    }
    return std::sqrt(v);
}
