//
// Created by jkonieczny on 14.06.15.
//

#ifndef METODY_MATRIX_H
#define METODY_MATRIX_H


#include "utils.h"
#include "Vec.h"

class Matrix {
    double *data;
    int w;
    int h;
public:
    Matrix(int w, int h);
    Matrix(int w);
    Matrix(const Matrix &m);
    Matrix& operator=(Matrix &m);
    static Matrix identity(int s);
    ~Matrix();

    double get(const int x, const int y);
    double operator()(const int x, const int y);
    void set(const int x, const int y, double val);

    Matrix reverse(bool isDiagonal=false);

    double det();

    void dump();

    Matrix transpose();

    Matrix operator+(Matrix &m);
    Matrix operator*(Matrix &m);
    Vec operator*(Vec &v);
    Matrix operator*(const double v);

    friend class Vec;
};


#endif //METODY_MATRIX_H
