#include <iostream>
#include <cstdio>
#include "Fun1.h"
#include "NR.h"

using namespace std;

int main() {
//
//    int s = 4;
//    Matrix m(s);
//    for(int x=0; x<s; x++)
//        for(int y=0; y<s; y++)
//            m.set(x,y,rand()%11+2);
//    printf("%lf\n", m.det());
//    printf("\n\n");

    Fun1 *f = new Fun1();
    Vec v(2);
    v.set(0, 0);
    v.set(1, 0);

    f->getGradient(v).dump();
    f->getHessan(v).dump();

    //NR n(f);
    //Vec x = n.getLowest(v);
    //x.dump();

    return 0;
}