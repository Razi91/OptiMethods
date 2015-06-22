//
// Created by jkonieczny on 15.06.15.
//

#ifndef METODY_VEC_H
#define METODY_VEC_H


class Vec {
    double *data;
    int size;

public:
    Vec(int size);
    ~Vec();

    const int getSize();
    const double get(int i);
    const double operator()(int i);
    Vec operator+(const Vec &b);
    Vec operator-(const Vec &b);
    Vec operator*(const Vec &b);
    Vec operator/(const Vec &b);
    Vec operator*(const double b);
    Vec operator/(const double b);

    double d();

    void set(int i, double v);

    const void dump();

    friend class Matrix;
};

Vec operator*(const double a, const Vec &b);

#endif //METODY_VEC_H
