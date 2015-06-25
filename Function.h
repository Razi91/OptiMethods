#ifndef METODY_FUNCTION_H
#define METODY_FUNCTION_H


#include <utility>
#include <stack>
#include "utils.h"
#include "Matrix.h"
#include "Vec.h"

class Function {
public:
    virtual double get(const Vec &x);
    virtual double operator()(const Vec &x) = 0;

    virtual Matrix getHessan(const Vec &x);
    virtual Vec getGradient(const Vec &x);
};


class Derivative : public Function {
    Function *f;
    unsigned p;
public:
    Derivative(Function *f, unsigned dir);
    virtual double operator()(const Vec &x);
};

#endif //METODY_FUNCTION_H
