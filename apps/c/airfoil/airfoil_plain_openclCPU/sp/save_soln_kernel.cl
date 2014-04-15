// user function definition in separate header file
//#include "save_soln.h"
inline void save_soln(
		__global const float * restrict q,
		__global       float * restrict qold){
  for (int n=0; n<4; n++) qold[n] = q[n];
}

/*
 * min / max definitions
 */

#ifndef MIN
#define MIN(a,b) ((a<b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a>b) ? (a) : (b))
#endif

// OpenCL kernel function
__kernel void op_opencl_save_soln(
		__global const float * restrict arg0,
		__global       float * restrict arg1,
		               int               set_size) {

	// process set elements
	int n = get_global_id(0);

	if(n<set_size) {
		save_soln( &arg0[n*4],
				       &arg1[n*4] );
	}
}
