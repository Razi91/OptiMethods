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

    double get(int i);
    double operator()(int i);
    void set(int i, double v);

    void dump();

    friend class Matrix;
};


#endif //METODY_VEC_H
