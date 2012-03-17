#ifndef __OP_OPENCL_RT_SUPPORT_H
#define __OP_OPENCL_RT_SUPPORT_H

#include "op_lib_cpp.h"
#include "op_rt_support.h"
#include "op_seq.h"

#include <CL/cl.h>

#define LOG_FATAL    (1)
#define LOG_ERR      (2)
#define LOG_WARN     (3)
#define LOG_INFO     (4)
#define LOG_DBG      (5)

#define DEBUG_LEVEL LOG_WARN

#ifdef DEBUG_LEVEL
#define LOG(level, ...) do {  \
                              if (level <= DEBUG_LEVEL ) { \
                                                                fprintf(stderr,"%s:%d:", __FILE__, __LINE__); \
                                                                fprintf(stderr, __VA_ARGS__); \
                                                                fprintf(stderr, "\n"); \
                                                                fflush(stderr); \
                                                            } \
                          } while (0)
#else
#define LOG(level, ...)  do { } while(0)
#endif

//the standard value for the warpsize is 1, as this code should also run on a CPU,
//which only has 1 thread executing in lockstep.
//in case you have OP_WARPSIZE > 1 and you run it on a CPU, and you get lots of nan,
//it's due to the warpsize.
#define OP_WARPSIZE 1

extern int OP_plan_index;
extern op_plan * OP_plans;

extern char *OP_consts_h, *OP_reduct_h;
extern cl_mem OP_consts_d, OP_reduct_d;

extern cl_context       cxGPUContext;
extern cl_command_queue cqCommandQueue;
extern cl_device_id     *cpDevice;
extern cl_uint          ciNumDevices;
extern cl_uint          ciNumPlatforms;
extern cl_platform_id   *cpPlatform;

#ifdef __cplusplus
extern "C" {
#endif

inline void assert_m( int val, const char* errmsg );

void compileProgram ( const char *filename );

inline void OpenCLDeviceInit( int argc, char **argv );

cl_kernel getKernel( const char *kernel_name );

cl_mem op_allocate_constant( void *buf, size_t size );

void op_mvHostToDevice( void **map, int size );

void op_cpHostToDevice( cl_mem *data_d, void **data_h, int size );

void releaseMemory ( cl_mem memobj );

void reallocConstArrays( int consts_bytes );

void reallocReductArrays( int reduct_bytes );

void mvConstArraysToDevice( int consts_bytes );

void mvReductArraysToDevice( int reduct_bytes );

void mvReductArraysToHost( int reduct_bytes );

cl_mem allocateSharedMemory ( size_t size );

#ifdef __cplusplus
}
#endif

#endif
