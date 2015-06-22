#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include "Function.h"
#include "utils.h"



Matrix Function::getHessan(const Vec &x) {
    return Matrix(1);
}
Vec Function::getGradient(const Vec &x) {
    return Vec(1);
}

Derivative::Derivative(Function *f, unsigned dir) {
    this->f = f;
    this->dir = dir;
}

double Derivative::operator()(const Vec &x, int p) {
    Function &fun = *f;
    const double delta = std::max(fabs(x(p) * 0.001), 0.001);
    const double fdelta = 1.0 / (delta * 2.0);
    Vec v = x;
    v.set(p, x(p) + delta);

    double xv = fun(x);
    v.set(p, x(p) - delta);
    xv -= fun(x);
    xv *= fdelta;
    return xv;
}

double Function::get(const double x, const double y) {
    return this->operator()(x, y);
}
double Function::get(const pt &p) {
    return this->operator()(p.x, p.y);
}
