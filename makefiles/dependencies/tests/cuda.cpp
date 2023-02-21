#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    auto err = cudaGetDeviceCount(&deviceCount);
    return err;
}
