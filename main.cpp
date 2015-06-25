#include <iostream>
#include <cstdio>
#include "Fun1.h"
#include "NR.h"
#include "Fun2.h"

using namespace std;

int main() {
//    int s = 4;
//    Matrix m(s);
//    for(int x=0; x<s; x++){
//        for(int y=0; y<s; y++)
//            m(x,y) = 1+rand() % 5;
//    }
//    m.dump();
//    Matrix i = m.inverse();
//    printf("%lf\n", m.det());
//    i.dump();
//    printf("%lf\n", i.det());

    Fun1 *f = new Fun1();
    Fun2 *f2 = new Fun2();
    Vec v(2);
    v.set(0, 0);
    v.set(1, 0);

    int S = 10;
    Vec l(S);
    for(int i=0; i<S; i++){
        l[i] = rand()%1000 / 10.0;
    }

    printf("Newton-Raphson\n");
    {
        NewtonRaphson n(f2);
        Vec x = n.getLowest(l);
        x.dump();
    }
    printf("Dawidon\n");
    {
        Davidon n(f2);
        Vec x = n.getLowest(l);
        x.dump();
    }

    return 0;
}