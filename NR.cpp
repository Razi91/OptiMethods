//
// Created by jkonieczny on 14.06.15.
//

#include <stdio.h>
#include <cmath>
#include "NR.h"

NR::NR(Fun1 &f) {
    this->f = f;
}

/*
 *
 * Algorytm:
 * x0
 * dk = - (H f(x_k))^-1 * g f(x_k)
 *
 */


// metoda Brenta


//double goldenRatio(pt &a, pt &b, Fun1 &f) {
//    double epsilon = 0.0001;
//    double k = (std::sqrt(5) - 1) / 2;
//    pt xL = b - k * (b - a);
//    pt xR = a + k * (b - a);
//    while ((b - a) > epsilon) {
//        if (f.get(xL) < f.get(xR)) {
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
//}
/*\
 * H - jednostkowa
 * iteracja M
 * gamma - różnica gradientów g - g(-1)
 *
 */

pt NR::getLowest(pt p) {
    pt np = p;
    double epsilon = 0.0001;
    const int max_iter = 10;
    int i = 0;
    while (i < max_iter) {
        Matrix *H = f.getHessan(p.x, p.y);
        Vec *g = f.getGradient(p.x, p.y);
        Matrix *Hp = H->reverse();
        Vec *E = Hp->mul(g);
        i++;
        np += pt(-E->get(0), -E->get(1));
        delete H;
        delete g;
        delete E;
        if (p.len(np) < epsilon)
            return np;
        p = np;
    }
    return p;
}
