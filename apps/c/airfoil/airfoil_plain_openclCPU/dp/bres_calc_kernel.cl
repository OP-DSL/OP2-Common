#define ROUND_UP(bytes) (((bytes) + 15) & ~15)
#define ZERO_double 0.0f

inline void bres_calc(__global double *x1, __global double *x2, __global double *q1,
                      __global double *adt1, double *res1, __global int *bound, double gm1, double eps, __constant double* qinf) {
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
  __global double *ind_arg0,
  __global double *ind_arg1,
  __global double *ind_arg2,
  __global double *ind_arg3,
  __global int   *ind_arg0_map_data,
  __global int   *ind_arg2_map_data,
  __global int   *arg5,
           int    block_offset,
  __global int   *blkmap,
  __global int   *offset,
  __global int   *nelems,
  __global int   *ncolors,
  __global int   *colors,
           int    nblocks,
           int    set_size,
  __constant double* gm1,
  __constant double* eps,
  __constant double* qinf) {

  double arg4_l[4];

//  __global int* __local ind_arg0_map;
//  __local int ind_arg0_size;
//  __global int* __local ind_arg1_map;
//  __local int ind_arg1_size;
//  __global int* __local ind_arg2_map;
//  __local int ind_arg2_size;
//  __global int* __local ind_arg3_map;
//  __local int ind_arg3_size;
//  __local double* __local ind_arg0_s;
//  __local double* __local ind_arg1_s;
//  __local double* __local ind_arg2_s;
//  __local double* __local ind_arg3_s;
  __local int    nelems2, ncolor;
  __local int    nelem, offset_b;

  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;
  if (get_local_id(0) == 0) {

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];

    nelem    = nelems[blockId];
    offset_b = offset[blockId];

    nelems2  = get_local_size(0)*(1+(nelem-1)/get_local_size(0));
    ncolor   = ncolors[blockId];

//    ind_arg0_size = ind_arg_sizes[0+blockId*4];
//    ind_arg1_size = ind_arg_sizes[1+blockId*4];
//    ind_arg2_size = ind_arg_sizes[2+blockId*4];
//    ind_arg3_size = ind_arg_sizes[3+blockId*4];
//
//    ind_arg0_map = &ind_map[0*set_size] + ind_arg_offs[0+blockId*4];
//    ind_arg1_map = &ind_map[2*set_size] + ind_arg_offs[1+blockId*4];
//    ind_arg2_map = &ind_map[3*set_size] + ind_arg_offs[2+blockId*4];
//    ind_arg3_map = &ind_map[4*set_size] + ind_arg_offs[3+blockId*4];
//
//    // set shared memory pointers
//
//    int nbytes = 0;
//    ind_arg0_s = (__local double *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg0_size*sizeof(double)*2);
//    ind_arg1_s = (__local double *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg1_size*sizeof(double)*4);
//    ind_arg2_s = (__local double *) &shared[nbytes];
//    nbytes    += ROUND_UP(ind_arg2_size*sizeof(double)*1);
//    ind_arg3_s = (__local double *) &shared[nbytes];
  }

  barrier(CLK_LOCAL_MEM_FENCE); // make sure all of above completed


  // copy indirect datasets into shared memory or zero increment

//  for (int n=get_local_id(0); n<ind_arg0_size*2; n+=get_local_size(0))
//    ind_arg0_s[n] = ind_arg0[n%2+ind_arg0_map[n/2]*2];
//
//  for (int n=get_local_id(0); n<ind_arg1_size*4; n+=get_local_size(0))
//    ind_arg1_s[n] = ind_arg1[n%4+ind_arg1_map[n/4]*4];
//
//  for (int n=get_local_id(0); n<ind_arg2_size*1; n+=get_local_size(0))
//    ind_arg2_s[n] = ind_arg2[n%1+ind_arg2_map[n/1]*1];
//
//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3_s[n] = ZERO_double;
//
//  barrier(CLK_LOCAL_MEM_FENCE);
  int map0idx;
  int map1idx;
  int map2idx;

  // process set elements

  for (int n=get_local_id(0); n<nelems2; n+=get_local_size(0)) {
    int col2 = -1;

    if (n<nelem) {

      // initialise local variables

      for (int d=0; d<4; d++)
        arg4_l[d] = ZERO_double;

      map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
      map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
      map2idx = ind_arg2_map_data[n + offset_b + set_size * 0];

      // user-supplied kernel call


      bres_calc(  ind_arg0 + map0idx*2, /* *x1    */
                  ind_arg0 + map1idx*2, /* *x2    */
                  ind_arg1 + map2idx*4, /* *q1    */
                  ind_arg2 + map2idx*1, /* *adt1  */
                  arg4_l,               /* *res1  */
                  arg5+(n+offset_b)*1,  /* *bound */
                  *gm1,                                       
                  *eps,                                      
                  qinf);

      col2 = colors[n+offset_b];
    }

    // store local variables

 //     int arg4_map;

 //     if (col2>=0) {
 //       arg4_map = arg_map[4*set_size+n+offset_b];
 //     }

    for (int col=0; col<ncolor; col++) {
      if (col2==col) {
        for (int d=0; d<4; d++)
          ind_arg3[d+map2idx*4] += arg4_l[d];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  }

  // apply pointered write/increment

//  for (int n=get_local_id(0); n<ind_arg3_size*4; n+=get_local_size(0))
//    ind_arg3[n%4+ind_arg3_map[n/4]*4] += ind_arg3_s[n];

}
