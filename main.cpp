#include <iostream>
#include <cstdio>
#include <omp.h>
#include <assert.h>
#include "Fun1.h"
#include "NR.h"
#include "Fun2.h"

using namespace std;


void testMatrix(){
    int s=3;
    Matrix m(s);
    m.set(0,0,7);
    m.set(1,0,9);
    m.set(2,0,3);
    m.set(0,1,1);
    m.set(1,1,2);
    m.set(2,1,4);
    m.set(0,2,5);
    m.set(1,2,8);
    m.set(2,2,7);
    m.dump();
    Matrix inv = m.inverse();
    inv.dump();
}

void testVector(){
    int s = 5;
    Vec v(s);
    for(int i=0; i<s; i++)
        v[i] = ((i+1)*2*3*7*13)&15;
    v.dump("");
    v.transpose().dump("T");
    (v*v.transpose()).dump("SQR");
}

int main(int argc, char *argv[]) {
//    testMatrix();
//    testVector();
//    return 0;
    Fun1 *f = new Fun1();
    Fun2 f2;
    Vec v(2);
    v.set(0, 0);
    v.set(1, 0);

    int S = 3;
    if (argc == 2){
        S = atoi(argv[1]);
    }
    Vec l(S);
    for (int i = 0; i < S; i++) {
        l[i] = i+4&0x16;//rand()%1000 / 10.0;
    }

    printf("\n\t\tNewton-Raphson\n");
    {
        double start_time = omp_get_wtime();
        Vec x = NewtonRaphson(f2, l);
        x.dump();
        printf("f(x) = %lf\n", f2(x));
        double time = omp_get_wtime() - start_time;
        printf("TIME: %lf\n", time);
    }
//    printf("\n\t\tDavidon\n");
//    {
//        double start_time = omp_get_wtime();
//        Vec x = Davidon(f2, l);
//        x.dump();
//        printf("f(x) = %lf\n", f2(x));
//        double time = omp_get_wtime() - start_time;
//        printf("TIME: %lf\n", time);
//    }
    return 0;
}