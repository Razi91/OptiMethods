#ifndef METODY_FUNCTION_H
#define METODY_FUNCTION_H


#include <utility>
#include <stack>
#include "utils.h"
#include "Matrix.h"
#include "Vec.h"

class Function {
public:

    virtual double get(const Vec &x) = 0;
    virtual double operator()(const Vec &x) = 0;


    virtual Matrix getHessan(const Vec &x);
    virtual Vec getGradient(const Vec &x);
};

class Derivative : public Function {
    Function *f;
    unsigned dir;
public:
    Derivative(Function *f, unsigned dir);
};

#endif //METODY_FUNCTION_H
