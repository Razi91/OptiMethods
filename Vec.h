//
// Created by jkonieczny on 15.06.15.
//

#ifndef METODY_VEC_H
#define METODY_VEC_H

class Matrix;

#define ALIGN 2

class Vec {
    double *data __attribute__ ((aligned(16))) = nullptr;
    int size;

public:
    Vec();
    Vec(const int size);
    Vec(const Vec &v);
    Vec& operator=(const Vec &v);
    ~Vec();

    int getSize() const;

    const double *getData();

    double get(const int i) const;
    double & operator[](const int i) const;
    double & v() const;

    void set(int i, double v);

    Matrix transpose();

    Vec operator-();

    Vec operator+(const Vec &b) const;
    Vec operator-(const Vec &b) const;
    Vec operator*(const Vec &b) const;
    Vec operator/(const Vec &b) const;


    Vec operator*(const double b) const;
    Vec operator/(const double b) const;


    Vec& operator+=(const Vec &b);
    Vec& operator-=(const Vec &b);
    Vec& operator*=(const Vec &b);
    Vec& operator/=(const Vec &b);
    Vec& operator*=(const double b);
    Vec& operator/=(const double b);

    Matrix operator*(const Matrix &b);

    Vec reverse();

    double len() const;

    const void dump(const char *str="");
    friend class Matrix;


    static Vec k_mul_a_minus_b(double k, const Vec &a, const Vec &b);
    static Vec k_mul_a_plus_b(double k, const Vec &a, const Vec &b);
};


Vec operator*(const double a, const Vec &b);
Vec operator/(const double a, const Vec &b);



#endif //METODY_VEC_H
