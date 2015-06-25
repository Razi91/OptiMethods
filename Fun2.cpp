//
// Created by jkonieczny on 25.06.15.
//

#include "Fun2.h"


#ifdef USE_CUDA
#include <cuda.h>

#endif

double Fun2::operator()(const Vec &x) {
    double v = 0;
    int s = x.getSize();
    if (s == 2){
        return 100*(x[1]-x[0])*(x[1]-x[0]) + (1-x[0])*(1-x[0]);
    }
    #ifdef USE_CUDA
    double y[s];

    cudaMalloc( (void**)&y, s*sizeof(double) );
	//cudaMalloc( (void**)&bd, s*sizeof(double) );
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
	//cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );

    for(int i=0; i<s-1; i++)
        v += y[i];


    #else
    double p1, p2;
    #pragma omp parallel for private(p1, p2) reduction(+: v)
    for(int i=0; i<s-1; i++){
        p1 = x[i+1] - x[i]*x[i];
        p1 *= p1;
        p1 *= 100;
        p2 = 1- x[i];
        p2 *= p2;
        v = v + p1 + p2;
    }
    // 100*(y-x)^2 + (1-x)^2
    // 1;1
    #endif
    return v;
}
