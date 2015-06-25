//
// Created by jkonieczny on 14.06.15.
//

#ifndef METODY_MATRIX_H
#define METODY_MATRIX_H

#include "Vec.h"

class Matrix {
    double *data;
    int w;
    int h;
public:
    Matrix() : Matrix(1,1) {};
    Matrix(const int w, const int h);
    Matrix(const int w);
    Matrix(const Matrix &m);
    Matrix& operator=(const Matrix &m);
    ~Matrix();

    static Matrix identity(const int s);

    double get(const int x, const int y) const;
    double & operator()(const int x, const int y) const;
    void set(const int x, const int y, double val);

    Matrix inverse(bool isDiagonal = false) const;

    double det() const;

    void dump() const;

    Matrix transpose() const;

    Matrix operator+(const Matrix &m);
    Matrix operator-(const Matrix &m);
    Matrix operator*(const Matrix &m);
    Matrix operator/(const Matrix &m);
    Vec operator*(const Vec &v);
    Matrix operator*(const double v);
    Matrix operator/(const double v);

    friend class Vec;
};


#endif //METODY_MATRIX_H
