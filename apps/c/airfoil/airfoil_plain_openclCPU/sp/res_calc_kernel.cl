//#include "res_calc.h"

#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ZERO_float 0.0f

inline void res_calc(
		__global const float * restrict x1,
		__global const float * restrict x2,
		__global const float * restrict q1,
		__global const float * restrict q2,
    __global const float * restrict adt1,
    __global const float * restrict adt2,
                   float * restrict res1,
                   float * restrict res2,
//        __global float * restrict res1,
//        __global float * restrict res2,
        float gm1,
        float eps) {

  float dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri   = 1.0f/q1[0];
  p1   = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));
  vol1 =  ri*(q1[1]*dy - q1[2]*dx);

  ri   = 1.0f/q2[0];
  p2   = gm1*(q2[3]-0.5f*ri*(q2[1]*q2[1]+q2[2]*q2[2]));
  vol2 =  ri*(q2[1]*dy - q2[2]*dx);

  mu = 0.5f*((*adt1)+(*adt2))*eps;

  f = 0.5f*(vol1* q1[0]         + vol2* q2[0]        ) + mu*(q1[0]-q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f*(vol1* q1[1] + p1*dy + vol2* q2[1] + p2*dy) + mu*(q1[1]-q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f*(vol1* q1[2] - p1*dx + vol2* q2[2] - p2*dx) + mu*(q1[2]-q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(q2[3]+p2)    ) + mu*(q1[3]-q2[3]);
  res1[3] += f;
  res2[3] -= f;
}

// OpenCL kernel function
__kernel void op_opencl_res_calc(
  __global const float* restrict ind_arg0,
  __global const float* restrict ind_arg1,
  __global const float* restrict ind_arg2,
  __global       float  *ind_arg3,
  __global const int   * restrict ind_arg0_map_data,
  __global const int   * restrict ind_arg1_map_data,
  __global const int   * restrict ind_arg2_map_data,
  __global const int   * restrict ind_arg3_map_data,
                 int    block_offset,
  __global const int   * restrict blkmap,
  __global const int   * restrict offset,
  __global const int   * restrict nelems,
  __global const int   * restrict ncolors,
  __global const int   * restrict colors,
           int    nblocks,
           int    set_size,
  __constant float* gm1,
  __constant float* eps) {

  float arg6_l[4];
  float arg7_l[4];
  int    ncolor;
  int    nelem, offset_b;

  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;

    // get sizes and shift pointers and direct-mapped data
    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
    nelem       = nelems[blockId];
    offset_b    = offset[blockId];
    ncolor      = ncolors[blockId];
    const int n = get_local_id(0);

    int col2 = -1;

    int map0idx;
    int map1idx;
    int map2idx;
    int map3idx;

    if (n<nelem) {

      //initialise local variables
      for ( int d=0; d<4; d++ ){
        arg6_l[d] = ZERO_float;
      }
      for ( int d=0; d<4; d++ ){
        arg7_l[d] = ZERO_float;
      }

      map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
      map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
      map2idx = ind_arg2_map_data[n + offset_b + set_size * 0];
      map3idx = ind_arg2_map_data[n + offset_b + set_size * 1];

      // user-supplied kernel call
      res_calc(
        &ind_arg0[2 * map0idx], /* *x1    */
        &ind_arg0[2 * map1idx], /* *x2    */
        &ind_arg1[4 * map2idx], /* *q1    */
        &ind_arg1[4 * map3idx], /* *q2    */
        &ind_arg2[1 * map2idx], /* *adt1  */
        &ind_arg2[1 * map3idx], /* *adt2  */
        arg6_l,                   /* *res1  */
        arg7_l,                   /* *res2  */
        *gm1,
        *eps);
//      res_calc(
//        &ind_arg0[2 * map0idx], /* *x1    */
//        &ind_arg0[2 * map1idx], /* *x2    */
//        &ind_arg1[4 * map2idx], /* *q1    */
//        &ind_arg1[4 * map3idx], /* *q2    */
//        &ind_arg2[1 * map2idx], /* *adt1  */
//        &ind_arg2[1 * map3idx], /* *adt2  */
//        &ind_arg3[map2idx*4], /* *res1  */
//        &ind_arg3[map3idx*4], /* *res2  */
//        *gm1,
//        *eps);
      col2 = colors[n+offset_b];
    } // end if

    //store local variables - colored increments
    for ( int col=0; col<ncolor; col++ ){
      if (col2==col) {
        for ( int d=0; d<4; d++ ){
          //AtomicAdd(&ind_arg3[d+map2idx*4], arg6_l[d]);
          //AtomicAdd(&ind_arg3[d+map3idx*4], arg7_l[d]);
          ind_arg3[d+map2idx*4] += arg6_l[d];
          ind_arg3[d+map3idx*4] += arg7_l[d];
        }
      }
      //barrier(CLK_GLOBAL_MEM_FENCE);//__syncthreads();
    }
}
