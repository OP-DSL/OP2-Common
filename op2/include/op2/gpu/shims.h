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
#define gpuStreamNonBlocking cudaStreamNonBlocking
#define gpuEventDisableTiming cudaEventDisableTiming

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
#define gpuMemsetAsync cudaMemsetAsync

#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define gpuHostMalloc cudaMallocHost
#define gpuHostFree cudaFreeHost

#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStreamSynchronize cudaStreamSynchronize

#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize

#define gpuGetSymbolAddress cudaGetSymbolAddress

#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties

#define gpuFuncGetAttributes cudaFuncGetAttributes
#define gpuFuncSetAttribute cudaFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

// Driver API
#define gpuDrvResult_t CUresult
#define gpuDrvModule_t CUmodule
#define gpuDrvFunction_t CUfunction
#define gpuDrvDevicePtr_t CUdeviceptr

#define gpuDrvFuncSetAttribute cuFuncSetAttribute
#define gpuDrvFuncAttributeMaxDynamicSharedMemorySize CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define gpuDrvFuncGetAttribute cuFuncGetAttribute
#define gpuDrvFuncAttribute_t CUfunction_attribute
#define gpuDrvFuncAttributeNumRegs CU_FUNC_ATTRIBUTE_NUM_REGS
#define gpuDrvFuncAttributeMaxThreadsPerBlock CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define gpuDrvFuncAttributeLocalSizeBytes CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define gpuDrvFuncAttributeSharedSizeBytes CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES

#define gpuDrvGetErrorName cuGetErrorName
#define gpuDrvModuleLoadData cuModuleLoadData
#define gpuDrvModuleUnload cuModuleUnload
#define gpuDrvModuleGetFunction cuModuleGetFunction
#define gpuDrvModuleGetGlobal cuModuleGetGlobal
#define gpuDrvMemcpyHtoDAsync cuMemcpyHtoDAsync
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
#define gpuStreamNonBlocking hipStreamNonBlocking
#define gpuEventDisableTiming hipEventDisableTiming

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
#define gpuMemsetAsync hipMemsetAsync

#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define gpuHostMalloc hipHostMalloc
#define gpuHostFree hipHostFree

#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamSynchronize hipStreamSynchronize

#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize

#define gpuGetSymbolAddress hipGetSymbolAddress

#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties

#define gpuFuncGetAttributes hipFuncGetAttributes
#define gpuFuncSetAttribute hipFuncSetAttribute
#define gpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

// Driver API
#define gpuDrvResult_t hipError_t
#define gpuDrvModule_t hipModule_t
#define gpuDrvFunction_t hipFunction_t
#define gpuDrvDevicePtr_t hipDeviceptr_t

// HIP has no attribute setter for a module function, and an AMD device needs
// no opt-in for dynamic shared memory.
#define gpuDrvFuncSetAttribute(function, attribute, value) hipSuccess
#define gpuDrvFuncGetAttribute hipFuncGetAttribute
#define gpuDrvFuncAttribute_t hipFunction_attribute
#define gpuDrvFuncAttributeNumRegs HIP_FUNC_ATTRIBUTE_NUM_REGS
#define gpuDrvFuncAttributeMaxThreadsPerBlock HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define gpuDrvFuncAttributeLocalSizeBytes HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
#define gpuDrvFuncAttributeSharedSizeBytes HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES

#define gpuDrvGetErrorName hipGetErrorString
#define gpuDrvModuleLoadData hipModuleLoadData
#define gpuDrvModuleUnload hipModuleUnload
#define gpuDrvModuleGetFunction hipModuleGetFunction
#define gpuDrvModuleGetGlobal hipModuleGetGlobal
// Before ROCm 7 the source is a non-const pointer, only read.
#define gpuDrvMemcpyHtoDAsync(dst, src, bytes, stream) \
    hipMemcpyHtoDAsync(dst, const_cast<void *>(src), bytes, stream)
#define gpuDrvLaunchKernel hipModuleLaunchKernel

// RTC API
#define gpuRtcResult_t hiprtcResult
#define gpuRtcProgram_t hiprtcProgram

#define gpuRtcGetErrorString hiprtcGetErrorString
// Before ROCm 7 the headers and their names are non-const, only read.
#define gpuRtcCreateProgram(prog, src, name, count, headers, names) \
    hiprtcCreateProgram(prog, src, name, count, const_cast<const char **>(headers), \
                        const_cast<const char **>(names))
#define gpuRtcCompileProgram hiprtcCompileProgram
#define gpuRtcGetProgramLogSize hiprtcGetProgramLogSize
#define gpuRtcGetProgramLog hiprtcGetProgramLog
#define gpuRtcGetCodeSize hiprtcGetCodeSize
#define gpuRtcGetCode hiprtcGetCode
#define gpuRtcDestroyProgram hiprtcDestroyProgram

#endif
