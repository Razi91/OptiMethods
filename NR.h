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

Vec goldenRatio(Vec a, Vec b, Function &f);


Vec NewtonRaphson(Function &f, Vec x) {
    Vec np = x;
    double epsilon = 0.001;
    int max_iter = 5;
    Matrix H(x.getSize());
    Matrix H1(x.getSize());
    Vec g = f.getGradient(x);
//    g.dump("g");
    while (max_iter--) {
        H = f.getHessan(x);
        H1 = H.inverse();
        g = f.getGradient(x);
//        H.dump("H");
//        H1.dump("H-1");
//        g.dump("g");
        Vec d = -(H1 * g);
        Vec x_i = goldenRatio(x, x + d, f);
        //Vec x_i = x + d;
        Vec g_i = f.getGradient(x_i);
        if ((x_i - x).len() < epsilon) {
            return x_i;
        }
        x = x_i;
    }
    return x;
}

Vec Davidon(Function &f, const Vec &p0) {
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
    Vec d[n + 1];
    Vec tmp(p0.getSize());
    // 1
    x[0] = p0;
    H[0] = Matrix::identity(p0.getSize());
    g[0] = f.getGradient(x[0]);
    i = 1;
    while (true) {
//        x[i-1].dump("x[i-1]");
//        g[i-1].dump("g[i-1]");
//        printf("==%d\n", i);
        // 2
        d[i - 1] = -(H[i - 1] * g[i - 1]);
        // 3 minimalizacja
        tmp = x[i - 1] + d[i - 1];
        x[i] = goldenRatio(x[i - 1], tmp, f);
        alpha = x[i] - x[i - 1];
        alphaT = alpha.transpose();
        // 4
        g[i] = f.getGradient(x[i]);
        // 5
        double f1 = f(x[i]);
        double f2 = f(x[i - 1]);
        epsilon = f1 * 0.0001;
        if (fabs(f1 - f2) < epsilon && alpha.len() < epsilon*100) {
//            x[i].dump("x[i]");
//            d[i - 1].dump("d[i-1]");
//            x[i - 1].dump("x[i-1]");
//            tmp.dump("tmp");
            printf("diff  %lf -- %lf\n", f1, f2);
//            alpha.dump("alpha");
            return x[i];
        }
        gamma = g[i] - g[i - 1];
        gammaT = gamma.transpose();
        Matrix m1 = (alpha * alphaT) / (alphaT * gamma).v();
        Matrix m2 = (H[i - 1] * gamma * gammaT * H[i - 1]) / (gammaT * H[i - 1] * gamma).v();
        // 6
        H[i] = H[i - 1] + m1 + m2;
//        if (i == n) {
//            H[i-1] = H[0];
//            //H[0] = H[i];
//            i = 1;
//        } else {
//            H[i - 1] = H[i];
//            g[i - 1] = g[i];
//            i++;
//        }
        //alternatywnie
        H[i-1] = H[i];
        x[i-1] = x[i];
        g[i-1] = g[i];
    }
}

Vec goldenRatio(Vec a, Vec b, Function &f) {
    double epsilon = 0.001;
    double k = (std::sqrt(5) - 1) / 2;
    Vec xL = b - k * (b - a);
    Vec xR = a + k * (b - a);
    double L;
    double R;
    while ((b - a).len() > epsilon) {
        L = f(xL);
        R = f(xR);
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
