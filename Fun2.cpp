//
// Created by jkonieczny on 25.06.15.
//

#include <omp.h>
#include <stdio.h>
#include "Fun2.h"


#ifdef USE_CUDA
#include <cuda.h>

#endif

//double Fun2::operator()(const Vec &x) {
//    double v = 0;
//    int s = x.getSize();
//    if (s == 2){
//        return 100*(x[1]-x[0])*(x[1]-x[0]) + (1-x[0])*(1-x[0]);
//    }
//#ifdef USE_CUDA
//    double y[s];
//
//    cudaMalloc( (void**)&y, s*sizeof(double) );
//	//cudaMalloc( (void**)&bd, s*sizeof(double) );
//	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice );
//	//cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice );
//
//    for(int i=0; i<s-1; i++)
//        v += y[i];
//
//
//    #else
//    double p1, p2;
//#pragma omp parallel for private(p1, p2) reduction(+: v)
//    for(int i=0; i<s-1; i++){
//        p1 = x[i+1] - x[i]*x[i];
//        p1 *= p1;
//        p1 *= 100;
//        p2 = 1- x[i];
//        p2 *= p2;
//        v = v + p1 + p2;
//    }
//    // 100*(y-x)^2 + (1-x)^2
//    // 1;1
//#endif
//    return v;
//}

double Fun2::operator()(const Vec &x) {
    double v = 0;
    double v2 = 0;
    int s = x.getSize();
//    if (s == 2) {
//        return (x[0] - 3) * (x[0] - 3) + (x[1] - 4) * (x[1] - 4) + (x[0] - x[1] + 1) * (x[0] - x[1] + 1);
//    }
    double p1, p2, p3, p4;
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
//            printf("p1s\n");
            #pragma omp parallel for private(p1, p2) reduction(+: v)
            for (int i = 0; i < s; i++) {
                p1 = (x[i]-(2+i+1));
                p1 *= p1;
                p2 = (2*(i+1)-1)*p1;
                v += p2;
            }

            int nthreads = omp_get_num_threads();
//            printf("Number of threads = %d\n",nthreads);
//            printf("p1e\n");
        }
        #pragma omp section
        {
//            printf("p2s\n");
            #pragma omp parallel for private(p3, p4) reduction(+: v2)
            for (int i = 0; i < s-1; i++) {
                p4 = 0;
                for (int j = i+1; j < s; j++) {
                    p3 = (x[i]-x[j]+((j+1)-(i+1)));
                    p3 *= p3;
                    p4 += p3;
                }
                v2 += p4;
            }
//            printf("p2e\n");
        }
    }
//    printf("v=%lf\n", v);
    return v+v2;
}
