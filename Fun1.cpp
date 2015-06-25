
#include "Fun1.h"
#include <cmath>
#include <omp.h>

double Fun1::operator()(const Vec &x) {
    return (x[0]-2)*(x[0]-2) + (x[1]+1)*(x[1]+1);
}