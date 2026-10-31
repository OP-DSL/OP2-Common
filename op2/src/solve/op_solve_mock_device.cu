// The mock library's arithmetic on the device (mock.h).

#include "mock.h"

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>

namespace op::solve::mock {

namespace {

constexpr int block = 256;

unsigned blocks(int n) { return (unsigned)((n + block - 1) / block); }

__global__ void spmv_kernel(Format f, int d, int n, Part local, Part nonlocal, const double *x, const double *halo,
                            double *y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = row_product(f, d, local, i, x) + (nonlocal.rows > 0 ? row_product(f, d, nonlocal, i, halo) : 0.0);
}

__global__ void inverse_diagonal_kernel(Format f, int d, int n, Part local, double *inv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        inv[i] = 1.0 / diagonal(f, d, local, i);
}

__global__ void dot_kernel(int n, const double *a, const double *b, double *sum) {
    __shared__ double partial[block];
    double value = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        value += a[i] * b[i];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int s = block / 2; s > 0; s /= 2) {
        if ((int)threadIdx.x < s)
            partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(sum, partial[0]);
}

__global__ void axpy_kernel(int n, double alpha, const double *x, double *y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += alpha * x[i];
}

__global__ void xpay_kernel(int n, const double *x, double beta, double *y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + beta * y[i];
}

__global__ void multiply_kernel(int n, const double *a, const double *b, double *y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] * b[i];
}

} // namespace

double *allocate(std::size_t count) {
    double *memory = nullptr;
    GPU_CHECK(gpuMalloc((void **)&memory, sizeof(double) * count + 1));
    GPU_CHECK(gpuMemset(memory, 0, sizeof(double) * count + 1));
    return memory;
}

void free(void *memory) { GPU_CHECK(gpuFree(memory)); }

void to_host(void *dst, const void *src, std::size_t bytes) {
    GPU_CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToHost));
}

void from_host(void *dst, const void *src, std::size_t bytes) {
    GPU_CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyHostToDevice));
}

void spmv(const Format& f, int d, int n, const Part& local, const Part& nonlocal, const double *x, const double *halo,
          double *y) {
    if (n > 0)
        spmv_kernel<<<blocks(n), block>>>(f, d, n, local, nonlocal, x, halo, y);
    GPU_CHECK(gpuGetLastError());
}

void inverse_diagonal(const Format& f, int d, int n, const Part& local, double *inv) {
    if (n > 0)
        inverse_diagonal_kernel<<<blocks(n), block>>>(f, d, n, local, inv);
    GPU_CHECK(gpuGetLastError());
}

double dot(int n, const double *a, const double *b) {
    static double *sum = nullptr;
    if (sum == nullptr)
        GPU_CHECK(gpuMalloc((void **)&sum, sizeof(double)));
    GPU_CHECK(gpuMemset(sum, 0, sizeof(double)));
    if (n > 0)
        dot_kernel<<<blocks(n) < 1024 ? blocks(n) : 1024, block>>>(n, a, b, sum);
    GPU_CHECK(gpuGetLastError());
    double value;
    GPU_CHECK(gpuMemcpy(&value, sum, sizeof(double), gpuMemcpyDeviceToHost));
    return value;
}

void axpy(int n, double alpha, const double *x, double *y) {
    if (n > 0)
        axpy_kernel<<<blocks(n), block>>>(n, alpha, x, y);
    GPU_CHECK(gpuGetLastError());
}

void xpay(int n, const double *x, double beta, double *y) {
    if (n > 0)
        xpay_kernel<<<blocks(n), block>>>(n, x, beta, y);
    GPU_CHECK(gpuGetLastError());
}

void multiply(int n, const double *a, const double *b, double *y) {
    if (n > 0)
        multiply_kernel<<<blocks(n), block>>>(n, a, b, y);
    GPU_CHECK(gpuGetLastError());
}

} // namespace op::solve::mock
