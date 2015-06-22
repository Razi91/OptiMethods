
#ifndef METODY_FUN1_H
#define METODY_FUN1_H


#include "Function.h"
#include "Vec.h"

class Fun1 : public Function {
public:
    virtual double operator()(Vec &x);

};


#endif //METODY_FUN1_H
