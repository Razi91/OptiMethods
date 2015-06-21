#include <iostream>
#include <cstdio>
#include "Fun1.h"
#include "NR.h"

using namespace std;

int main() {

    int s = 4;
    Matrix m(s);
    for(int x=0; x<s; x++)
        for(int y=0; y<s; y++)
            m.set(x,y,rand()%11+2);
    printf("%lf\n", m.det());


    return 0;
}