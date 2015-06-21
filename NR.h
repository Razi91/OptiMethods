//
// Created by jkonieczny on 14.06.15.
//

#ifndef METODY_NR_H
#define METODY_NR_H


#include "Function.h"
#include "utils.h"
#include "Fun1.h"

class NR {
    Fun1 f;
public:
    NR(Fun1 &f);
    pt getLowest(pt p);
};


#endif //METODY_NR_H
