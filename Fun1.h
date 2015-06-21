
#ifndef METODY_FUN1_H
#define METODY_FUN1_H


#include "Function.h"
#include "Vec.h"

class Fun1 : public Function {
public:
    virtual double operator()(const double x, const double y);

    virtual Matrix* getHessan(const double x, const double y);
    virtual Vec* getGradient(const double x, const double y);
};


#endif //METODY_FUN1_H
