//
// Created by jkonieczny on 14.06.15.
//

#include <stdio.h>
#include <cmath>
#include <assert.h>
#include <string.h>
#include "Matrix.h"
#include <cstdlib>
#include <algorithm>
#include <omp.h>

Matrix::Matrix(const int w, const int h) {
    this->w = w;
    this->h = h;
    this->data = new double[w * h];
}

Matrix::Matrix(const int w) {
    this->w = w;
    this->h = w;
    this->data = new double[w * w];
}

Matrix::~Matrix() {
    delete this->data;
}

Matrix::Matrix(const Matrix &m) {
    this->w = m.w;
    this->h = m.h;
    this->data = new double[w * h];
    memcpy(data, m.data, w * h * sizeof(double));
}

Matrix &Matrix::operator=(Matrix &&m) {
    this->~Matrix();
    this->w = m.w;
    this->h = m.h;
    this->data = new double[w * h];
    memcpy(data, m.data, w * h * sizeof(double));
    return *this;
}

double Matrix::get(const int x, const int y) const {
    return data[x + y * w];
}

double & Matrix::operator()(const int x, const int y) const {
    return data[x + y * w];
}


void Matrix::set(const int x, const int y, double val) {
    data[x + y * w] = val;
}


void Matrix::dump() const {
    printf("Matrix(%d)\n", w);
    for (int y = 0; y < h; y++) {
        printf("| ");
        for (int x = 0; x < w; x++) {
            printf("%3.3lf ", get(x, y));
        }
        printf(" |\n");
    }


}

double Matrix::det() const {
    assert(w == h);
    dump();
    if (w == 2) {
        return get(0, 0) * get(1, 1) - get(1, 0) * get(0, 1);
    } else if (w == 3) {
        return 0
               + get(0, 0) * get(1, 1) * get(2, 2)
               + get(0, 1) * get(1, 2) * get(2, 0)
               + get(0, 2) * get(1, 0) * get(2, 1)

               - get(0, 2) * get(1, 1) * get(2, 0)
               - get(0, 1) * get(1, 0) * get(2, 2)
               - get(0, 0) * get(1, 2) * get(2, 1);
    }
    double d = 0;
    for (int i = 0, ii = 1; i < w; i++, ii *= -1) {
        Matrix m(w - 1);
        for (int y = 0; y < w - 1; y++)
            for (int x = 0; x < w - 1; x++) {
                m.set(x, y, get(x + (x >= i), y + 1));
            }
        d += ii * get(i, 0) * m.det();
    }
    return d;
}

Matrix Matrix::reverse(bool isDiagonal)const {
    assert(w == h);
    Matrix m(w, h);
    if (isDiagonal) {
        std::fill(m.data, m.data + w * h, 0.0);
        int min = w < h ? w : h;
        for (int i = 0; i < min; i++) {
            m.set(i, i, 1.0 / get(i, i));
        }
    }
    if (std::fabs(det()) < 0.01)
        return Matrix(1);

    if (w == 1 && h == 1){
        m(0,0) = 1.0/get(0,0);
    }

    if (w == 2 && h == 2) {
        double ad = 1 / det();
        m.set(0, 0, ad * get(1, 1));
        m.set(1, 0, -ad * get(1, 0));
        m.set(0, 1, -ad * get(0, 1));
        m.set(1, 1, ad * get(0, 0));
    }

    return m;
}

Matrix Matrix::identity(const int s) {
    Matrix m(s, s);
    const double zero = 0.0;
    std::fill(m.data, m.data + s * s, zero);
    for (int i = 0; i < s; i++)
        m.set(i, i, 1);
    return m;
}

Matrix Matrix::operator+(const Matrix &m) {
    assert(w == m.w & h == m.h);
    Matrix n(m.w, h);
    for (int x = 0; x < m.w; x++) {
        for (int y = 0; y < h; y++) {
            n.set(x, y, get(x, y) + n.get(x, y));
        }
    }
    return n;
}
Matrix Matrix::operator-(const Matrix &m) {
    assert(w == m.w & h == m.h);
    Matrix n(m.w, h);
    for (int x = 0; x < m.w; x++) {
        for (int y = 0; y < h; y++) {
            n.set(x, y, get(x, y) - n.get(x, y));
        }
    }
    return n;
}

#ifdef USE_CUDA
#define TILE_WIDTH 2
__global__ void MatrixMul( double *Md , double *Nd , double *Pd , const int WIDTH )
{
    // calculate thread id
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
    for (int k = 0 ; k<WIDTH ; k++ )
    {
        Pd[row*WIDTH + col] += Md[row * WIDTH + k ] * Nd[ k * WIDTH + col] ;
    }
}
#endif

Matrix Matrix::operator*(const Matrix &m) {
    assert(w == m.h);
    Matrix n(m.w, h);
    if (omp_in_parallel()) {
#pragma omp parallel for
        for (int x = 0; x < m.w; x++) {
            for (int y = 0; y < h; y++) {
                double s = 0.0;
                for (int i = 0; i < w; i++) {
                    s += get(i, y) * m(x, i);
                }
                n.set(x, y, s);
            }
        }
    } else {
        for (int x = 0; x < m.w; x++) {
            for (int y = 0; y < h; y++) {
                double s = 0.0;
                for (int i = 0; i < w; i++) {
                    s += get(i, y) * m(x, i);
                }
                n.set(x, y, s);
            }
        }
    }
    return n;
}

Matrix Matrix::operator/(const Matrix &m) {
    return this->operator*(m.reverse());
}

Matrix Matrix::operator*(const double v) {
    Matrix m(w, h);
    if (!omp_in_parallel()) {
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                m.set(x, y, get(x, y) * v);
            }
        }
    } else {
#pragma omp parallel for
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                m.set(x, y, get(x, y) * v);
            }
        }
    }
    return m;
}
Matrix Matrix::operator/(const double v) {
    Matrix m(w, h);
    if (!omp_in_parallel()) {
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                m.set(x, y, get(x, y) / v);
            }
        }
    } else {
#pragma omp parallel for
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                m.set(x, y, get(x, y) / v);
            }
        }
    }
    return m;
}

Matrix Matrix::transpose() const {
    Matrix m(h, w);
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            m.set(y, x, get(x, y));
        }
    }
    return m;
}

Vec Matrix::operator*(const Vec &b) {
    assert(w == b.getSize());
    Vec v(h);
    if (!omp_in_parallel()) {
#pragma omp parallel for
        for (int y = 0; y < h; y++) {
            double s = 0;
            for (int i = 0; i < w; i++) {
                s += b[i] * get(i, y);
            }
            v.set(y, s);
        }

    } else {
        for (int y = 0; y < h; y++) {
            double s = 0;
            for (int i = 0; i < w; i++) {
                s += b[i] * get(i, y);
            }
            v.set(y, s);
        }
    }
    return v;
}
