#include <hip/hip_runtime.h>

int main() {
    int deviceCount = 0;
    auto err = hipGetDeviceCount(&deviceCount);
    return err;
}
