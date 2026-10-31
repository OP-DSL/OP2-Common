// The solver layer's device side: int arrays in device memory.

#include <op2/solve/solve.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>

namespace op::solve {

int *upload(const std::vector<int>& values) {
    int *device = nullptr;
    GPU_CHECK(gpuMalloc((void **)&device, sizeof(int) * values.size() + 1));
    GPU_CHECK(gpuMemcpy(device, values.data(), sizeof(int) * values.size(), gpuMemcpyHostToDevice));
    return device;
}

void release(void *memory) { GPU_CHECK(gpuFree(memory)); }

} // namespace op::solve
