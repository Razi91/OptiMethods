#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "Function.h"
#include "utils.h"


Matrix Function::getHessan(const Vec &x) {
    Matrix m(x.getSize());
    if (x.getSize() == 2) {
        Derivative dx(this, 0);
        Derivative dy(this, 1);
        Derivative dxx(&dx, 0);
        Derivative dxy(&dx, 1);
        Derivative dyx(&dy, 0);
        Derivative dyy(&dy, 1);
        #pragma omp parallel num_threads(4)
        {
            int i = omp_get_thread_num();
            if (i == 0)
                m.set(0, 0, dxx(x));
            if (i == 1)
                m.set(1, 0, dxy(x));
            if (i == 2)
                m.set(0, 1, dyx(x));
            if (i == 3)
                m.set(1, 1, dyy(x));
        }
        return m;
    }

    #pragma omp parallel for
    for (unsigned int px = 0; px < x.getSize(); px++) {
        Derivative d(this, px);
        for (unsigned int py = 0; py < x.getSize(); py++) {
            Derivative d2(&d, py);
            m.set(px, py, d2(x));
        }
    }
    return m;
}

Vec Function::getGradient(const Vec &x) {
    Vec v(x.getSize());
    #pragma omp parallel for
    for (unsigned int i = 0; i < x.getSize(); i++) {
        Derivative d(this, i);
        v.set(i, d(x));
    }
    return v;
}

/*
 * Klasa pochodnej
 */

Derivative::Derivative(Function *f, unsigned dir) {
    this->f = f;
    this->p = dir;
}


double Derivative::operator()(const Vec &x) {
    Function &fun = *f;
    const double delta = std::max(fabs(x[p] * 0.0001), 0.0001);
    const double fdelta = 1.0 / (delta * 2.0);
    Vec v = x;
    v.set(p, x[p] + delta);
    double xv = fun(v);
    v.set(p, x[p] - delta);
    xv -= fun(v);
    xv *= fdelta;
    return xv;
}

double Function::get(const Vec &x) {
    return this->operator()(x);
}
