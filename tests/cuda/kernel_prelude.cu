// The translated kernels' integer intrinsics and atomic increment on the
// device for a 64-bit integer, which offline is the host's int64_t.

#include <op2/rt/kernel/inc.h>
#include <op2/rt/kernel/prelude.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../test_support.h"

namespace k = op::rt::kernel;

namespace {

constexpr int threads = 64;

__global__ void intrinsics(int64_t a, int64_t b, int64_t *out, int64_t *sum) {
    if (threadIdx.x == 0) {
        out[0] = k::abs(a);
        out[1] = k::min(a, b);
        out[2] = k::max(a, b, int64_t{7}, int64_t{1}, int64_t{2});
        out[3] = k::mod(a, b);
        out[4] = k::copysign(b, a);
    }
    k::atomic_inc(*sum, a);
}

} // namespace

int main() {
    int64_t *out = nullptr;
    int64_t *sum = nullptr;
    CHECK(cudaMallocManaged(&out, 5 * sizeof(int64_t)) == cudaSuccess);
    CHECK(cudaMallocManaged(&sum, sizeof(int64_t)) == cudaSuccess);
    *sum = 0;

    intrinsics<<<1, threads>>>(-5, 3, out, sum);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);

    CHECK(out[0] == 5 && out[1] == -5 && out[2] == 7 && out[3] == -2 && out[4] == -3);
    CHECK(*sum == -5 * threads);

    cudaFree(out);
    cudaFree(sum);
    std::printf("kernel prelude tests passed\n");
    return EXIT_SUCCESS;
}
