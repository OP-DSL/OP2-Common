#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ZERO_double 0.0

inline void bres_calc(
		__global const double * restrict x1,
		__global const double * restrict x2,
		__global const double * restrict q1,
    __global const double * restrict adt1,
             double       * restrict res1,
    __global const int    * restrict bound,
             double                  gm1,
             double                  eps,
    __constant double * qinf) {
  double dx,dy,mu, ri, p1,vol1, p2,vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f/q1[0];
  p1 = gm1*(q1[3]-0.5f*ri*(q1[1]*q1[1]+q1[2]*q1[2]));

  if (*bound==1) {
    res1[1] += + p1*dy;
    res1[2] += - p1*dx;
  }
  else {
    vol1 =  ri*(q1[1]*dy - q1[2]*dx);

    ri   = 1.0f/qinf[0];
    p2   = gm1*(qinf[3]-0.5f*ri*(qinf[1]*qinf[1]+qinf[2]*qinf[2]));
    vol2 =  ri*(qinf[1]*dy - qinf[2]*dx);

    mu = (*adt1)*eps;

    f = 0.5f*(vol1* q1[0]         + vol2* qinf[0]        ) + mu*(q1[0]-qinf[0]);
    res1[0] += f;
    f = 0.5f*(vol1* q1[1] + p1*dy + vol2* qinf[1] + p2*dy) + mu*(q1[1]-qinf[1]);
    res1[1] += f;
    f = 0.5f*(vol1* q1[2] - p1*dx + vol2* qinf[2] - p2*dx) + mu*(q1[2]-qinf[2]);
    res1[2] += f;
    f = 0.5f*(vol1*(q1[3]+p1)     + vol2*(qinf[3]+p2)    ) + mu*(q1[3]-qinf[3]);
    res1[3] += f;
  }
}

// OpenCL kernel function

__kernel void op_opencl_bres_calc(
  __global double * restrict ind_arg0,
  __global double * restrict ind_arg1,
  __global double * restrict ind_arg2,
  __global double * restrict ind_arg3,
  __global int    * restrict ind_arg0_map_data,
  __global int    * restrict ind_arg2_map_data,
  __global int    * restrict arg5,
           int               block_offset,
  __global int    * restrict blkmap,
  __global int    * restrict offset,
  __global int    * restrict nelems,
  __global int    * restrict ncolors,
  __global int    * restrict colors,
           int               nblocks,
           int               set_size,
  __constant double* gm1,
  __constant double* eps,
  __constant double* qinf) {

  double arg4_l[4];

  int ncolor, nelem, offset_b;

  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;
    // get sizes and shift pointers and direct-mapped data
    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
    nelem    = nelems[blockId];
    offset_b = offset[blockId];
    ncolor   = ncolors[blockId];

  int map0idx;
  int map1idx;
  int map2idx;

  // process set elements
  int n = get_local_id(0);
    int col2 = -1;

    if (n<nelem) {

      // initialise local variables
      for (int d=0; d<4; d++)
        arg4_l[d] = ZERO_double;

      map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
      map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
      map2idx = ind_arg2_map_data[n + offset_b + set_size * 0];

      // user-supplied kernel call
      bres_calc( &ind_arg0[map0idx*2], /* *x1    */
                 &ind_arg0[map1idx*2], /* *x2    */
                 &ind_arg1[map2idx*4], /* *q1    */
                 &ind_arg2[map2idx*1], /* *adt1  */
                  arg4_l,               /* *res1  */
                 &arg5[(n+offset_b)*1],  /* *bound */
                  *gm1,                                       
                  *eps,                                      
                  qinf);

      col2 = colors[n+offset_b];
    }

    // apply pointered write/increment
    for (int col=0; col<ncolor; col++) {
      if (col2==col) {
        for (int d=0; d<4; d++)
          ind_arg3[d+map2idx*4] += arg4_l[d];
      }
//      barrier(CLK_LOCAL_MEM_FENCE);
    }
}
