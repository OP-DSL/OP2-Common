#include "op_lib_cpp.h"
#include "op_rt_support.h"

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

#ifdef __cplusplus
extern "C" {
#endif

inline void assert_m( int val, const char* errmsg );

void compileProgram ( const char *filename );

inline void OpenCLDeviceInit( int argc, char **argv );

void op_mvHostToDevice( void **map, int size );

void op_cpHostToDevice( cl_mem *data_d, void **data_h, int size );

//void op_fetch_data( op_dat dat );
//void op_init( int argc, char **argv, int diags );

//op_dat op_decl_dat_char( op_set set, int dim, char const *type,
//                        int size, char *data, char const *name );

//op_plan *op_plan_get( char const *name, op_set set, int part_size,
//                     int nargs, op_arg *args, int ninds, int *inds );

//void op_exit( );

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
