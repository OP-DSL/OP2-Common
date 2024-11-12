#pragma once

#ifdef OP2_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

// Macros
#define GPU_SUCCESS CUDA_SUCCESS
#define GPURTC_SUCCESS NVRTC_SUCCESS

// Runtime API
#define gpuStream_t cudaStream_t
#define gpuEvent_t cudaEvent_t
#define gpuError_t cudaError_t
#define gpuDeviceProp_t cudaDeviceProp
#define gpuFuncAttributes_t cudaFuncAttributes

#define gpuSuccess cudaSuccess
#define gpuFuncCachePreferL1 cudaFuncCachePreferL1
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuHostRegisterDefault cudaHostRegisterDefault

#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuPeekAtLastError cudaPeekAtLastError

#define gpuLaunchKernel cudaLaunchKernel

#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemset cudaMemset

#define gpuMallocAsync cudaMallocAsync
#define gpuFreeAsync cudaFreeAsync
#define gpuMemcpyAsync cudaMemcpyAsync

#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDefault cudaMemcpyDefault

#define gpuHostMalloc cudaMallocHost
#define gpuHostFree cudaFreeHost
#define gpuHostRegister cudaHostRegister
#define gpuHostUnregister cudaHostUnregister

#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStreamSynchronize cudaStreamSynchronize

#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent

#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuGetSymbolAddress cudaGetSymbolAddress

#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuDeviceSetCacheConfig cudaDeviceSetCacheConfig
#define gpuGetDeviceProperties cudaGetDeviceProperties

#define gpuFuncGetAttributes cudaFuncGetAttributes

// Driver API
#define gpuDrvResult_t CUresult
#define gpuDrvModule_t CUmodule
#define gpuDrvFunction_t CUfunction

#define gpuDrvGetErrorName cuGetErrorName
#define gpuDrvModuleLoadData cuModuleLoadData
#define gpuDrvModuleGetFunction cuModuleGetFunction
#define gpuDrvOccupancyMaxPotentialBlockSize cuOccupancyMaxPotentialBlockSize
#define gpuDrvLaunchKernel cuLaunchKernel

// RTC API
#define gpuRtcResult_t nvrtcResult
#define gpuRtcProgram_t nvrtcProgram

#define gpuRtcGetErrorString nvrtcGetErrorString
#define gpuRtcCreateProgram nvrtcCreateProgram
#define gpuRtcCompileProgram nvrtcCompileProgram
#define gpuRtcGetProgramLogSize nvrtcGetProgramLogSize
#define gpuRtcGetProgramLog nvrtcGetProgramLog
#define gpuRtcGetCodeSize nvrtcGetCUBINSize
#define gpuRtcGetCode nvrtcGetCUBIN
#define gpuRtcDestroyProgram nvrtcDestroyProgram

#endif

#ifdef OP2_HIP

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

// Macros
#define GPU_SUCCESS hipSuccess
#define GPURTC_SUCCESS HIPRTC_SUCCESS

// Runtime API
#define gpuStream_t hipStream_t
#define gpuEvent_t hipEvent_t
#define gpuError_t hipError_t
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuFuncAttributes_t hipFuncAttributes

#define gpuSuccess hipSuccess
#define gpuFuncCachePreferL1 hipFuncCachePreferL1
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuHostRegisterDefault hipHostRegisterDefault

#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuPeekAtLastError hipPeekAtLastError

#define gpuLaunchKernel hipLaunchKernel

#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemset hipMemset

#define gpuMallocAsync hipMallocAsync
#define gpuFreeAsync hipFreeAsync
#define gpuMemcpyAsync hipMemcpyAsync

#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDefault hipMemcpyDefault

#define gpuHostMalloc hipHostMalloc
#define gpuHostFree hipHostFree
#define gpuHostRegister hipHostRegister
#define gpuHostUnregister hipHostUnregister

#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamSynchronize hipStreamSynchronize

#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent

#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuGetSymbolAddress hipGetSymbolAddress

#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuDeviceSetCacheConfig hipDeviceSetCacheConfig
#define gpuGetDeviceProperties hipGetDeviceProperties

#define gpuFuncGetAttributes hipFuncGetAttributes

// Driver API
#define gpuDrvResult_t hipError_t
#define gpuDrvModule_t hipModule_t
#define gpuDrvFunction_t hipFunction_t

#define gpuDrvGetErrorName hipGetErrorString
#define gpuDrvModuleLoadData hipModuleLoadData
#define gpuDrvModuleGetFunction hipModuleGetFunction
#define gpuDrvOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define gpuDrvLaunchKernel hipModuleLaunchKernel

// RTC API
#define gpuRtcResult_t hiprtcResult
#define gpuRtcProgram_t hiprtcProgram

#define gpuRtcGetErrorString hiprtcGetErrorString
#define gpuRtcCreateProgram hiprtcCreateProgram
#define gpuRtcCompileProgram hiprtcCompileProgram
#define gpuRtcGetProgramLogSize hiprtcGetProgramLogSize
#define gpuRtcGetProgramLog hiprtcGetProgramLog
#define gpuRtcGetCodeSize hiprtcGetCodeSize
#define gpuRtcGetCode hiprtcGetCode
#define gpuRtcDestroyProgram hiprtcDestroyProgram

#endif
