#ifndef METODY_FUNCTION_H
#define METODY_FUNCTION_H


#include <utility>
#include <stack>
#include "utils.h"
#include "Matrix.h"
#include "Vec.h"

enum Dir { X=0, Y=1 };

class Function {
public:

    virtual double operator()(const double x, const double y) = 0;
    double get(const double x, const double y);
    double get(const pt &p);
    double operator()(pt p);

    virtual Matrix* getHessan(const double x, const double y);
    virtual Vec* getGradient(const double x, const double y);

    Matrix* getHessan(const pt &y);
    Vec* getGradient(const pt &y);
};

class Derivative : public Function {
    Function *f;
public:

    Derivative(Function *f, Dir dir);

    virtual double operator()(double x, double y);
protected:

    Dir dir;

};


#endif //METODY_FUNCTION_H
