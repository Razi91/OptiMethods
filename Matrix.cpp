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

const bool optomp = false;

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

Matrix &Matrix::operator=(const Matrix &m) {
    if (h != m.h || w != m.w) {
        delete data;
        this->w = m.w;
        this->h = m.h;
        this->data = new double[w * h];
    }
    memcpy(data, m.data, w * h * sizeof(double));
    return *this;
}

double Matrix::get(const int x, const int y) const {
    return data[x + y * w];
}

double &Matrix::operator()(const int x, const int y) const {
    return data[x + y * w];
}


void Matrix::set(const int x, const int y, double val) {
    data[x + y * w] = val;
}


void Matrix::dump(const char *str) const {
    printf("Matrix(%d) %s\n", w, str);
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
    if (w == 1)
        return get(0, 0);
    else if (w == 2) {
        return get(0, 0) * get(1, 1) - get(1, 0) * get(0, 1);
    }
    else if (w == 3) {
        return 0
               + get(0, 0) * get(1, 1) * get(2, 2)
               + get(0, 1) * get(1, 2) * get(2, 0)
               + get(0, 2) * get(1, 0) * get(2, 1)

               - get(0, 2) * get(1, 1) * get(2, 0)
               - get(0, 1) * get(1, 0) * get(2, 2)
               - get(0, 0) * get(1, 2) * get(2, 1);
    }
    double d = 0;
    if (omp_in_parallel()) {
        for (int i = 0, ii = 1; i < w; i++, ii *= -1) {
            Matrix m(w - 1);
            for (int y = 0; y < w - 1; y++)
                for (int x = 0; x < w - 1; x++) {
                    m.set(x, y, get(x + (x >= i), y + 1));
                }
            d += ii * get(i, 0) * m.det();
        }

    }
    else {
#pragma omp parallel for reduction(+: d)
        for (int i = 0; i < w; i++) {
            Matrix m(w - 1);
            for (int y = 0; y < w - 1; y++)
                for (int x = 0; x < w - 1; x++) {
                    m.set(x, y, get(x + (x >= i), y + 1));
                }
            d += i & 1 ? -1 : 1 * get(i, 0) * m.det();
        }
    }
    return d;
}

int GetMinor(double **src, double **dest, int row, int col, int order) {
    // indicate which col and row is being copied to dest
    int colCount = 0, rowCount = 0;

    for (int i = 0; i < order; i++) {
        if (i != row) {
            colCount = 0;
            for (int j = 0; j < order; j++) {
                // when j is not the element
                if (j != col) {
                    dest[rowCount][colCount] = src[i][j];
                    colCount++;
                }
            }
            rowCount++;
        }
    }

    return 1;
}

// Calculate the determinant recursively.
double CalcDeterminant(double **mat, int order) {
    if (order == 1)
        return mat[0][0];
    double det = 0;
    double **minor;
    minor = new double *[order - 1];
    for (int i = 0; i < order - 1; i++)
        minor[i] = new double[order - 1];

    for (int i = 0; i < order; i++) {
        GetMinor(mat, minor, 0, i, order);
        det += (i % 2 == 1 ? -1.0 : 1.0) * mat[0][i] * CalcDeterminant(minor, order - 1);
    }
    for (int i = 0; i < order - 1; i++)
        delete[] minor[i];
    delete[] minor;
    return det;
}

Matrix Matrix::inverse(bool isDiagonal) const {
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

    if (w == 1 && h == 1) {
        m(0, 0) = 1.0 / get(0, 0);
    } else if (w == 2 && h == 2) {
        double ad = 1 / det();
        m.set(0, 0, ad * get(1, 1));
        m.set(1, 0, -ad * get(1, 0));
        m.set(0, 1, -ad * get(0, 1));
        m.set(1, 1, ad * get(0, 0));
    } else { // dowolna
        double det = 1.0 / this->det();
        double **A = new double *[w];
        for (int i = 0; i < w; i++) {
            A[i] = new double[w];
            for (int j = 0; j < w; j++)
                A[i][j] = get(i, j);
        }
        // memory allocation
        double *temp = new double[(w - 1) * (w - 1)];
        double **minor = new double *[w - 1];
        for (int i = 0; i < w - 1; i++)
            minor[i] = temp + (i * (w - 1));
//        #pragma omp parallel for
        for (int j = 0; j < w; j++) {
            for (int i = 0; i < w; i++) {
                GetMinor(A, minor, j, i, w);
                m.set(i, j, det * CalcDeterminant(minor, w - 1));
                if ((i + j) % 2 == 1)
                    m.set(i, j, -m.get(i, j));
            }
        }
        //delete [] minor[0];
        delete[] temp;
        delete[] minor;
        delete[] A;
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
    int x;
#pragma omp parallel for
    for (x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            set(x, y, m.get(x, y) + get(x, y));
        }
    }
    return n;
}

Matrix Matrix::operator-(const Matrix &m) {
    assert(w == m.w & h == m.h);
    Matrix n(m.w, h);
#pragma omp parallel for
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
    if (omp_in_parallel() && optomp) {
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
#pragma omp parallel for num_threads(4)
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
    return this->operator*(m.inverse());
}

Matrix Matrix::operator*(const double v) {
    Matrix m(w, h);
    if (omp_in_parallel() && optomp) {
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
    if (omp_in_parallel() && optomp) {
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
    if (omp_in_parallel() && optomp) {
        for (int y = 0; y < h; y++) {
            double s = 0;
            for (int i = 0; i < w; i++) {
                s += b[i] * get(i, y);
            }
            v.set(y, s);
        }
    } else {
#pragma omp parallel for
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
