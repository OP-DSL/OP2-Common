#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

//
// OpenCL specific data structure
//

typedef struct{
  cl_platform_id*   platform_id;
  cl_device_id    device_id;
  cl_device_id*  subdev_id;
  cl_uint      n_devices;
  cl_uint      n_platforms;
  cl_command_queue  command_queue;
  cl_kernel*     kernel;
  cl_program     program;
  cl_context     context;
  cl_uint   n_kernels;
  cl_mem*   constant;
  cl_uint  n_constants;
//  cl_mem*    data_d; // cl_mem struct corresponding to op_core_dat char* data_d
} op_opencl_core;


