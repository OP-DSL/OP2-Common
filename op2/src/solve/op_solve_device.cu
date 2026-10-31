// The solver layer's device side: the permuting copy a staged commit runs.

#include <op2/solve/solve.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>

namespace op::solve {

namespace {

__global__ void scatter_kernel(double *dst, const double *src, const int *to, std::size_t n) {
    std::size_t k = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n)
        dst[to[k]] = src[k];
}

} // namespace

void scatter(double *dst, const double *src, const int *to, std::size_t n) {
    if (n == 0)
        return;
    const int block = 256;
    scatter_kernel<<<(unsigned)((n + block - 1) / block), block>>>(dst, src, to, n);
    GPU_CHECK(gpuGetLastError());
}

int *upload(const std::vector<int>& values) {
    int *device = nullptr;
    GPU_CHECK(gpuMalloc((void **)&device, sizeof(int) * values.size() + 1));
    GPU_CHECK(gpuMemcpy(device, values.data(), sizeof(int) * values.size(), gpuMemcpyHostToDevice));
    return device;
}

void release(void *memory) { GPU_CHECK(gpuFree(memory)); }

} // namespace op::solve
