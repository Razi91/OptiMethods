//
// Created by jkonieczny on 14.06.15.
//

#ifndef METODY_NR_H
#define METODY_NR_H


#include <stdio.h>
#include <cmath>
#include "NR.h"
#include "Function.h"
#include "utils.h"
#include "Fun1.h"

Vec goldenRatio(Vec &a, Vec &b, Fun1 &f);

class NR {
    Function *f;
public:
    NR(Function *f) {
        this->f = f;
    };

    Vec getLowest(Vec &x) {
        Vec np = x;
        double epsilon = 0.0001;
        const int max_iter = 50;
        int i = 0;
        //Matrix H = Matrix::identity(x.getSize());
        while (i < max_iter) {
//            Matrix &&H = f->getHessan(x);
//            Vec &&g = f->getGradient(x);
//            Matrix &&Hp = H.reverse();
//            //Vec&& E = Hp * g;
//            //Vec E = H.reverse()
//            i++;
//            np = np + E;
//            if ((x - np).d() < epsilon)
//                return np;
//            x = np;
        }
        return x;
    }
};

class NR2 {
    Function *f;
public:
    NR2(Function *f) {
        this->f = f;
    };

    Vec getLowest(Vec &x) {
        Vec np = x;
        double epsilon = 0.0001;
        const int max_iter = 50;
        int i = 0;
        Matrix H = Matrix::identity(x.getSize());
        while (i < max_iter) {
            Matrix A(x.getSize());
            Vec B(x.getSize());
            Vec g = f->getGradient(x);
            Vec di = -(H * g);
            Vec xn = x + di; //TODO: minfun
            Matrix dit = di.transpose();

            Vec gi = f->getGradient(x);
            Vec alpha = xn - x;
            Matrix alphaT = alpha.transpose();

            Vec gamma = gi - g;

            Matrix m1 = (alpha * alphaT) * (alphaT * gamma).reverse();
            //H = H + (dit * di) * ((dit * A * di).reverse()) * B;
            i++;
        }
        return x;
    }
};


/*
 *
 * Algorytm:
 * x0
 * dk = - (H f(x_k))^-1 * g f(x_k)
 *
 */


// metoda Brenta


Vec goldenRatio(Vec &a, Vec &b, Fun1 &f) {
    double epsilon = 0.001;
    double k = (std::sqrt(5) - 1) / 2;
    Vec xL = b - k * (b - a);
//    Vec xR = a + k * (b - a);
//    while ((b - a).d() > epsilon) {
//        if (f(xL) < f(xR)) {
//            b = xR;
//            xR = xL;
//            xL = b - k * (b - a);
//        }
//        else {
//            a = xL;
//            xL = xR;
//            xR = a + k * (b - a);
//        }
//    }
//    return (a + b) / 2;
}


#endif //METODY_NR_H
