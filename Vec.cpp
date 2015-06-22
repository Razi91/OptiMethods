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
        printf("%lf ", get(i));
    }
    printf("]\n");
}

int Vec::getSize() {
    return size;
}

Vec Vec::operator-() {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = -get(i);
    return v;
}
//////////////////////////////////////////////
Vec Vec::operator+(Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (get(i) + b[i]);
    return v;
}

Vec Vec::operator-(Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (get(i) - b[i]);
    return v;
}

Vec Vec::operator*(Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (get(i) * b[i]);
    return v;
}

Vec Vec::operator/(Vec &b) {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (get(i) / b[i]);
    return v;
}
Vec Vec::operator*(const double b) {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (get(i) * b);
    return v;
}

Vec operator*(double a, Vec &b) {
    return b * a;
}

Vec Vec::operator/(const double b) {
    Vec v(size);
    for (int i = 0; i < size; i++) {
        v.data[i] = (get(i) / b);
    }
    return v;
}




///////////////////////////////////////////////////
Vec Vec::operator+=(Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (get(i) + b[i]);
    return *this;
}

Vec Vec::operator-=(Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (get(i) - b[i]);
    return *this;
}

Vec Vec::operator*=(Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (get(i) * b[i]);
    return *this;
}

Vec Vec::operator/=(Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (get(i) / b[i]);
    return *this;
}
///////////////////////////////////////////////////////

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
