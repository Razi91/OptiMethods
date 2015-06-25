__global__ void fun_compute( double *x, double *y , const int N )
{
    int i = threadIdx.x;
    double p1, p2;
    if (i<N-1){
        p1 = x[i+1] - x[i]*x[i];
        p1 *= p1;
        p1 *= 100;
        p2 = 1- x[i];
        p2 *= p2;
        y[i] = p1 + p2;
    }
}