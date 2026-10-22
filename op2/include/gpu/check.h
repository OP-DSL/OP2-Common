#pragma once

// Stop the program on a failed runtime, driver or runtime-compiler call,
// naming the call and where it was made.

#include <gpu/shims.h>

#include <cstdio>
#include <cstdlib>

#define GPU_CHECK(x)                                                            \
    do {                                                                        \
        gpuError_t result = x;                                                  \
        if (result != gpuSuccess) {                                             \
            const char *msg = gpuGetErrorString(result);                        \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#ifdef OP2_CUDA
#define GPU_DRV_CHECK(x)                                                        \
    do {                                                                        \
        gpuDrvResult_t result = x;                                              \
        if (result != GPU_SUCCESS) {                                            \
            const char *msg;                                                    \
            gpuDrvGetErrorName(result, &msg);                                   \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#else
// HIP's driver calls are runtime calls.
#define GPU_DRV_CHECK(x) GPU_CHECK(x)
#endif

#define GPU_RTC_CHECK(x)                                                        \
    do {                                                                        \
        gpuRtcResult_t result = x;                                              \
        if (result != GPURTC_SUCCESS) {                                         \
            const char *msg = gpuRtcGetErrorString(result);                     \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
