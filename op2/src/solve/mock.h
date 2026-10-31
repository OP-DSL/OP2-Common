#pragma once

// The mock library's storage and arithmetic (op_solve_mock.cpp): a stand-in
// for a solver library that allocates its matrix in its own layout, local and
// non-local parts apart.  Host versions in op_solve_mock.cpp, device ones in
// op_solve_mock_device.cu, each in the space the variant's loops use.

#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OP2_MOCK_HD __host__ __device__
#else
#define OP2_MOCK_HD
#endif

namespace op::solve::mock {

// Block CSR (CompactBlockStorage) or block ELL (BELLStorage); a block's
// entries row or column major; a block ELL's blocks contiguous or each entry
// interleaved across the block rows.
struct Format {
    bool ell = false;
    bool column = false;
    bool interleaved = false;
};

// One part of a matrix as the mock holds it: rows block rows of d x d blocks,
// block CSR (ptr, col) or block ELL (width, col at k * rows + I, -1 padding).
struct Part {
    int rows = 0;
    int width = 0;
    const int *ptr = nullptr;
    const int *col = nullptr;
    double *values = nullptr;
};

// Where entry (r, c) of block k of block row I is: k counts the part's blocks
// in block CSR, the row's in block ELL.
OP2_MOCK_HD inline std::size_t index(const Format& f, int d, int rows, int I, int k, int r, int c) {
    const std::size_t e = f.column ? (std::size_t)c * d + r : (std::size_t)r * d + c;
    if (!f.ell)
        return (std::size_t)k * d * d + e;
    if (f.interleaved)
        return ((std::size_t)k * d * d + e) * rows + I;
    return ((std::size_t)k * rows + I) * d * d + e;
}

// Row i's sum over part p of value * x[column], an array of the part's
// columns.
OP2_MOCK_HD inline double row_product(const Format& f, int d, const Part& p, int i, const double *x) {
    const int I = i / d, r = i % d;
    double sum = 0.0;
    if (f.ell) {
        for (int k = 0; k < p.width; k++) {
            const int J = p.col[(std::size_t)k * p.rows + I];
            if (J < 0)
                continue;
            for (int c = 0; c < d; c++)
                sum += p.values[index(f, d, p.rows, I, k, r, c)] * x[(std::size_t)J * d + c];
        }
    } else {
        for (int k = p.ptr[I]; k < p.ptr[I + 1]; k++)
            for (int c = 0; c < d; c++)
                sum += p.values[index(f, d, p.rows, I, k, r, c)] * x[(std::size_t)p.col[k] * d + c];
    }
    return sum;
}

// Row i's diagonal entry in the local part.
OP2_MOCK_HD inline double diagonal(const Format& f, int d, const Part& p, int i) {
    const int I = i / d, r = i % d;
    if (f.ell) {
        for (int k = 0; k < p.width; k++)
            if (p.col[(std::size_t)k * p.rows + I] == I)
                return p.values[index(f, d, p.rows, I, k, r, r)];
    } else {
        for (int k = p.ptr[I]; k < p.ptr[I + 1]; k++)
            if (p.col[k] == I)
                return p.values[index(f, d, p.rows, I, k, r, r)];
    }
    return 0.0;
}

// In the loops' space: zeroed arrays, and host copies to and from them.
double *allocate(std::size_t count);
void free(void *memory);
void to_host(void *dst, const void *src, std::size_t bytes);
void from_host(void *dst, const void *src, std::size_t bytes);

// y = local x + nonlocal halo over n rows; inv[i] = 1 / the diagonal.
void spmv(const Format& f, int d, int n, const Part& local, const Part& nonlocal, const double *x, const double *halo,
          double *y);
void inverse_diagonal(const Format& f, int d, int n, const Part& local, double *inv);

// This rank's sum of a .* b; y += alpha x; y = x + beta y; y = a .* b.
double dot(int n, const double *a, const double *b);
void axpy(int n, double alpha, const double *x, double *y);
void xpay(int n, const double *x, double beta, double *y);
void multiply(int n, const double *a, const double *b, double *y);

} // namespace op::solve::mock
