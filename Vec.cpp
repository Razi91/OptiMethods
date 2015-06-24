//
// Created by jkonieczny on 15.06.15.
//

#include "Vec.h"
#include "Matrix.h"

#include <cassert>
#include <stdio.h>
#include <cmath>
#include <string.h>

Vec::Vec(const int size) {
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

Vec& Vec::operator=(const Vec &v){
    if ( size != v.size) {
        delete data;
        int x = size&1;
        data = new double[x?(size+16)&(~x):size];
    }
    memcpy(data, v.data, sizeof(double) * size);
    return *this;
}

double Vec::get(const int i) const {
    assert(i < size);
    return data[i];
}

double & Vec::operator[](const int i) const {
    assert(i < size);
    return data[i];
}
double & Vec::v() const {
    assert(size == 1);
    return data[0];
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

int Vec::getSize() const {
    return size;
}

Vec Vec::operator-() {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = -data[i];
    return v;
}
//////////////////////////////////////////////
Vec Vec::operator+(const Vec &b) const {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (data[i] + b.data[i]);
    return v;
}

Vec Vec::operator-(const Vec &b) const {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (data[i] - b.data[i]);
    return v;
}

Vec Vec::operator*(const Vec &b) const {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (data[i] * b.data[i]);
    return v;
}

Vec Vec::operator/(const Vec &b) const {
    assert(size == b.size);
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (data[i] / b.data[i]);
    return v;
}
Vec Vec::operator*(const double b) const {
    Vec v(size);
    for (int i = 0; i < size; i++)
        v.data[i] = (data[i] * b);
    return v;
}

Vec Vec::operator/(const double b) const {
    Vec v(size);
    for (int i = 0; i < size; i++) {
        v.data[i] = (data[i] / b);
    }
    return v;
}




///////////////////////////////////////////////////
Vec& Vec::operator+=(const Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (data[i] + b.data[i]);
    return *this;
}

Vec& Vec::operator-=(const Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (data[i] - b.data[i]);
    return *this;
}

Vec& Vec::operator*=(const Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (data[i] * b.data[i]);
    return *this;
}

Vec& Vec::operator/=(const Vec &b) {
    assert(size == b.size);
    for (int i = 0; i < size; i++)
        data[i] = (data[i] / b.data[i]);
    return *this;
}

Vec& Vec::operator*=(const double b) {
    for (int i = 0; i < size; i++)
        data[i] = (data[i] * b);
    return *this;
}

Vec& Vec::operator/=(const double b) {
    for (int i = 0; i < size; i++)
        data[i] = (data[i] / b);
    return *this;
}
///////////////////////////////////////////////////////
Matrix Vec::operator*(const Matrix &b) {
    assert(size == b.w & b.h == 1);
    Matrix m(size);
    for(int x=0; x<size; x++)
        for(int y=0; y<size; y++)
            m.set(x,y, b.get(x,0) * get(y));
    return m;
}

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
////
Vec operator*(const double a, const Vec &b) {
    return b*a;
}
Vec operator/(const double a, const Vec &b) {
    return b/a;
}