
#include "Fun1.h"
#include <cmath>
#include <omp.h>

double Fun1::operator()(const double x, const double y) {
    return (x-2)*(x-2) + (y+1)*(y+1);
}

/**
 * dx = 2x - 4
 * dy = 2(y + 1)
 * Hess:
 * [
 *    xx xy
 *    yx yy
 * ]
 *
 * [
 *  xx xy xz
 *  yx yy yz
 *  zx zy zz
 * ]
 *
 * min w 2;-1
 */
Matrix Fun1::getHessan(const Vec &x) {
    int s = x.getSize();
    Matrix m = new Matrix(s,s);
    #pragma omp parallel for
    for(int x=0; x<s; x++) {
        Derivative d(this, x);
        for(int y=0; y<s; x++) {
            Derivative d2(d, y);
            m.set(x,y, d2(x));
        }
    }
//    m->set(0,0, 2 );
//    m->set(1,0, 0 );
//    m->set(0,1, 0 );
//    m->set(1,1, 2 );
//    #pragma omp parallel num_threads(4)
//    {
//        int i = omp_get_thread_num();
//        if (i == 0){
//            Derivative dx(this, Dir::X);
//            Derivative dxx(&dx, Dir::X);
//            m->set(0,0, dxx(x,y));
//        }
//        if (i == 1 || omp_get_num_threads() < 2){
//            Derivative dx(this, Dir::X);
//            Derivative dxy(&dx, Dir::Y);
//            m->set(1,0, dxy(x,y));
//        }
//        if (i == 2 || omp_get_num_threads() < 3){
//            Derivative dy(this, Dir::X);
//            Derivative dyy(&dy, Dir::Y);
//            m->set(1,0, dyy(x,y));
//        }
//        if (i == 3 || omp_get_num_threads() < 4){
//            Derivative dy(this, Dir::Y);
//            Derivative dyy(&dy, Dir::Y);
//            m->set(1,1, dyy(x,y));
//        }
    }
    return m;
}

Vec * Fun1::getGradient(const Vec &x) {
    Vec *v = new Vec(2);
    #pragma omp parallel num_threads(2)
    {
        int i = omp_get_thread_num();
        if (i == 0){
            Derivative dx(this, Dir::X);
            v->set(0, dx(x,y));
        }
        if (i == 1 || omp_get_num_threads() != 2){
            Derivative dy(this, Dir::Y);
            v->set(1, dy(x,y));
        }
    }
    return v;
}