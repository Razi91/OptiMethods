#ifndef METODY_FUNCTION_H
#define METODY_FUNCTION_H


#include <utility>
#include <stack>
#include "utils.h"
#include "Matrix.h"
#include "Vec.h"

class Function {
public:
    virtual double get(Vec &x);
    virtual double operator()(Vec &x) = 0;

    virtual Matrix getHessan(Vec &x);
    virtual Vec getGradient(Vec &x);
};


class Derivative : public Function {
    Function *f;
    unsigned p;
public:
    Derivative(Function *f, unsigned dir);
    virtual double operator()(Vec &x);
};

#endif //METODY_FUNCTION_H
