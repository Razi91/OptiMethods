//
// Created by jkonieczny on 15.06.15.
//

#ifndef METODY_VEC_H
#define METODY_VEC_H

class Matrix;

class Vec {
    double *data __attribute__ ((aligned(16)));
    int size;

public:
    Vec(int size);

    Vec(const Vec &v);

    ~Vec();

    int getSize();

    double get(const int i);

    double operator[](const int i);
    void set(int i, double v);

    Matrix transpose();

    Vec operator-();



    Vec reverse();

    double d();



    const void dump();
    friend class Matrix;
};
Vec operator+(Vec &a, Vec &b);
Vec operator-(Vec &a, Vec &b);
Vec operator*(Vec &a, Vec &b);
Vec operator/(Vec &a, Vec &b);

Vec operator+=(Vec &a, Vec &b);
Vec operator-=(Vec &a, Vec &b);
Vec operator*=(Vec &a, Vec &b);
Vec operator/=(Vec &a, Vec &b);

Vec operator*(Vec &a, const double b);
Vec operator/(Vec &a, const double b);

Matrix operator*(Vec &a, Matrix &b);
Vec operator*(const double a, const Vec &b);

#endif //METODY_VEC_H
