//
// Created by jkonieczny on 25.06.15.
//

#ifndef METODY_FUN2_H
#define METODY_FUN2_H


#include "Function.h"

class Fun2 : public Function {

public:
    virtual double operator()(const Vec &x);
};


#endif //METODY_FUN2_H
