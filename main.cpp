#include <iostream>
#include <cstdio>
#include "Fun1.h"
#include "NR.h"

using namespace std;

int main() {

    Fun1 *f = new Fun1();
    Vec v(2);
    v.set(0, 0);
    v.set(1, 0);

    f->getGradient(v).dump();
    f->getHessan(v).dump();
    {
//        NewtonRaphson n(f);
//        Vec x = n.getLowest(v);
//        x.dump();
    }

    {
        Davidon n(f);
        Vec x = n.getLowest(v);
        x.dump();
    }

    return 0;
}