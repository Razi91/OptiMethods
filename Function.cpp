#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include "Function.h"
#include "utils.h"

double Function::operator()(pt p) {
    return this->operator()(p.x, p.y);
}

Matrix *Function::getHessan(const pt &p) {
    return getHessan(p.x, p.y);
}

Vec *Function::getGradient(const pt &y) {
    return getGradient(y.x, y.y);
}


Matrix * Function::getHessan(const double x, const double y) {
    return nullptr;
}
Vec * Function::getGradient(const double x, const double y) {
    return nullptr;
}

Derivative::Derivative(Function *f, Dir dir) {
    this->f = f;
    this->dir = dir;
}

double Derivative::operator()(double x, double y) {
    Function &fun = *f;
    const double deltax = std::max(fabs(x * 0.001), 0.001);
    const double fdeltax = 1.0 / (deltax * 2.0);
    const double deltay = std::max(fabs(y * 0.001), 0.001);
    const double fdeltay = 1.0 / (deltay * 2.0);
    switch (dir) {
        case X: {
            double xv = fun(x + deltax, y) - fun(x - deltax, y);
            xv *= fdeltax;
            return xv;
        }
        case Y: {
            double yv = fun(x, y + deltay) - fun(x, y - deltay);
            yv *= fdeltay;
            return yv;
        }
    }
}

double Function::get(const double x, const double y) {
    return this->operator()(x, y);
}
double Function::get(const pt &p) {
    return this->operator()(p.x, p.y);
}
