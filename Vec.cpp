//
// Created by jkonieczny on 15.06.15.
//

#include "Vec.h"
#include "Matrix.h"

#include <cassert>
#include <stdio.h>
#include <cmath>
#include <string.h>

Vec::Vec(int size) {
    int x = size&1;
    data = new double[x?(size+16)&(~x):size];
    this->size = size;
}

Vec::Vec(const Vec &v) {
    int size = v.size;
    int x = size&1;
    data = new double[x?(size+16)&(~x):size];
    this->size = v.size;
    memcpy(data, v.data, sizeof(double) * size);
}

Vec::~Vec() {
    delete data;
}

double Vec::get(const int i) {
    assert(i < size);
    return data[i];
}

double Vec::operator[](const int i) {
    assert(i < size);
    return data[i];
}

void Vec::set(int i, const double v) {
    assert(i < size);
    data[i] = v;
}

const void Vec::dump() {
    printf("Vector(%d)\n [", size);
    for (int i = 0; i < size; i++) {
        printf("%lf ", data[i]);
    }
    printf("]\n");
}

int Vec::getSize() {
    return size;
}

Vec Vec::operator-() {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = -data[i];
    return v;
}
//////////////////////////////////////////////
Vec operator+(Vec &a, Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (a.data[i] + b[i]);
    return v;
}

Vec operator-(Vec &a, Vec &b) {
    assert(a.size == b.size);
    Vec v(a.size);
    for (int i = 0; i < a.size; i++)
        v.data[i] = (a.data[i] - b[i]);
    return v;
}

Vec operator*(Vec &a, Vec &b) {
    assert(a.size == b.size);
    Vec v(a.size);
    for (int i = 0; i < a.size; i++)
        v.data[i] = (a.data[i] * b[i]);
    return v;
}

Vec operator/(Vec &a, Vec &b) {
    assert(a.size == b.size);
    Vec v(size);
    for (int i = 0; i < a.size; i++)
        v.data[i] = (a.data[i] / b[i]);
    return v;
}
Vec operator*(Vec &a, const double b) {
    Vec v(a.size);
    for (int i = 0; i < a.size; i++)
        v.data[i] = (a.data[i] * b);
    return v;
}

Vec operator*(double a, Vec &b) {
    return b * a;
}

Vec operator/(Vec &a, const double b) {
    Vec v(a.size);
    for (int i = 0; i < a.size; i++) {
        v.data[i] = (a.data[i] / b);
    }
    return v;
}




///////////////////////////////////////////////////
Vec operator+=(Vec &a, Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < a.size; i++)
        data[i] = (a.data[i] + b[i]);
    return a;
}

Vec operator-=(Vec &a, Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < a.size; i++)
        data[i] = (data[i] - b[i]);
    return a;
}

Vec operator*=(Vec &a, Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < a.size; i++)
        a.data[i] = (a.data[i] * b[i]);
    return a;
}

Vec operator/=(Vec &a, Vec &b) {
    assert(a.size == b.size);
    for (int i = 0; i < a.size; i++)
        a.data[i] = (a.data[i] / b[i]);
    return a;
}
///////////////////////////////////////////////////////

Vec Vec::reverse() {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = 1/data[i];
    return Vec(0);
}

double Vec::d() {
    double v = 0;
    for (int i = 0; i < size; i++) {
        v += get(i) * get(i);
    }
    return std::sqrt(v);
}

Matrix Vec::transpose() {
    Matrix m(size, 1);
    for(int i=0; i<size; i++)
        m.set(i, 0, data[i]);
    return m;
}


Matrix Vec::operator*(Matrix &b) {
    assert(size == b.w & b.h == 1);
    Matrix m(size);
    for(int x=0; x<size; x++)
        for(int y=0; y<size; y++)
            m.set(x,y, b.get(x,0) * get(y));
    return m;
}
