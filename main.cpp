#include <iostream>
#include <cstdio>
#include <omp.h>
#include "Fun1.h"
#include "NR.h"
#include "Fun2.h"

using namespace std;

int main(int argc, char *argv[]) {
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
    if (argc == 2){
        S = atoi(argv[1]);
    }
    Vec l(S);
    for (int i = 0; i < S; i++) {
        l[i] = i+4&0x16;//rand()%1000 / 10.0;
    }
    double start_time = omp_get_wtime();
//    volatile double s = 0;
//    for(volatile int z = 0; z<40000000; z++) {
//        volatile double ff = f2->get(v);
//        s += ff;
//    }
//    printf("f(x) = %lf\n", s);

    printf("Newton-Raphson\n");
    {
        NewtonRaphson n(f2);
        Vec x = n.getLowest(l);
        x.dump();
        printf("f(x) = %lf\n", f2->get(x));
    }
//    printf("Dawidon\n");
//    {
//        Davidon n(f2);
//        Vec x = n.getLowest(l);
//        x.dump();
//        printf("f(x) = %lf\n", f2->get(x));
//    }
    double time = omp_get_wtime() - start_time;
    printf("TIME: %lf\n", time);
    return 0;
}