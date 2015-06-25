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

Vec goldenRatio(Vec a, Vec b, Function *f);

class NewtonRaphson {
    Function *f;
public:
    NewtonRaphson(Function *f) {
        this->f = f;
    };

    Vec getLowest(Vec x) {
        Vec np = x;
        double epsilon = 0.001;
        int max_iter = 5;
        Matrix H(x.getSize());
        Matrix H1(x.getSize());
        Vec g = f->getGradient(x);
        while (max_iter--) {
            H = f->getHessan(x);
            H1 = H.inverse();
            g = f->getGradient(x);
            H.dump("H");
            H1.dump("H-1");
            g.dump();
            Vec d = -(H * g);
            Vec x_i = goldenRatio(x, x+d, f);
            //Vec x_i = x + d;
            Vec g_i = f->getGradient(x_i);
            if ((x_i - x).len() < epsilon) {
                return x_i;
            }
            x = x_i;
            x.dump();
        }
        printf("\nno\n");
        return x;
    }
};

class Davidon {
    Function *f;
public:
    Davidon(Function *f) {
        this->f = f;
    };

    Vec getLowest(const Vec &p0) {
        const int n = p0.getSize();
        int size = p0.getSize();
        double epsilon = 0.001;
        const int max_iter = 50;
        int i = 0;
        Vec alpha, gamma;
        Matrix alphaT, gammaT;
        Vec x[n + 1];
        Matrix H[n + 1];
        Vec g[n + 1];
        Vec tmp(p0.getSize());
        // 1
        x[0] = p0;
        H[0] = Matrix::identity(x[0].getSize());
        g[0] = f->getGradient(x[0]);
        i = 1;
        while (true) {
            // 2
            Vec di = -(H[i - 1] * g[i - 1]);
            // 3 minimalizacja
            //Vec x_n = x + di;
            tmp = x[i - 1] + di;
            x[i] = goldenRatio(x[i - 1], tmp, f);
            alpha = x[i] - x[i - 1];
            alphaT = alpha.transpose();
            // 4
            g[i] = f->getGradient(x[i - 1]);
            // 5
            if (alpha.len() < epsilon)
                return x[i];
            printf("len = %lf\n", alpha.len());
            gamma = g[i] - g[i - 1];
            gammaT = gamma.transpose();
            Matrix m1 = (alpha * alphaT) / (alphaT * gamma).v();
            Matrix m2 = (H[i - 1] * gamma * gammaT * H[i - 1]) / (gammaT * H[i - 1] * gamma).v();
            // 6
            H[i] = H[i - 1] + m1 + m2;
            x[i - 1] = x[i];
            g[i - 1] = g[i];
//            if (i == n && false) {
//                H[i-1] = H[0];
//                i = 1;
//            } else {
//                H[i - 1] = H[i];
//                g[i-1] = g[i];
//                //i++;
//            }
        }
    }
};

Vec goldenRatio(Vec a, Vec b, Function *f) {
    double epsilon = 0.001;
    double k = (std::sqrt(5) - 1) / 2;
    Vec xL = b - k * (b - a);
    Vec xR = a + k * (b - a);
    double L;
    double R;
    while ((b - a).len() > epsilon) {
        L = f->get(xL);
        R = f->get(xR);
        if (L < R) {
            b = xR;
            xR = xL;
            xL = b - Vec::k_mul_a_minus_b(k, b, a);
        }
        else {
            a = xL;
            xL = xR;
            xR = a + Vec::k_mul_a_minus_b(k, b, a);
        }
    }
    return (a + b) / 2;
}


#endif //METODY_NR_H
