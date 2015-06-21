//
// Created by jkonieczny on 15.06.15.
//

#include "Vec.h"

#include <cassert>
#include <stdio.h>

Vec::Vec(int size) {
    data = new double[size];
    this->size = size;
}

Vec::~Vec() {
    delete data;
}

double Vec::get(int i) {
    assert(i<size);
    return data[i];
}
double Vec::operator()(const int i) {
    return get(i);
}

void Vec::set(int i, double v) {
    assert(i<size);
    data[i] = v;
}

void Vec::dump() {
    printf("Vector(%d)\n [", size);
    for(int i=0; i<size; i++){
        printf("%lf ", get(i));
    }
    printf("]\n");
}
