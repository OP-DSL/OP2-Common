// OpenCL kernel function
//#include "adt_calc.h"

void adt_calc(__global const double * restrict x1, 
              __global const double * restrict x2, 
              __global const double * restrict x3, 
              __global const double * restrict x4, 
              __global const double * restrict q, 
              __global       double * restrict adt, 
              double gam, 
              double gm1, 
              double cfl){

  double dx,dy, ri,u,v,c;

  ri =  1.0f/q[0];
  u  =   ri*q[1];
  v  =   ri*q[2];
  c  = sqrt(gam*gm1*(ri*q[3]-0.5f*(u*u+v*v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy);

  *adt = (*adt) / cfl;
}

__kernel void op_opencl_adt_calc(
  __global const double * restrict ind_arg0, /* x */
  __global const int   * restrict ind_arg0_map_data,
  __global const double * restrict arg4,
  __global       double * restrict arg5, /* adt */
           const int              block_offset,
  __global const int   * restrict blkmap,
  __global const int   * restrict offset,
  __global const int   * restrict nelems,
  __global const int   * restrict ncolors,
  __global const int   * restrict colors,
           int    nblocks,
           int    set_size,
  __constant double *gam,
  __constant double *gm1,
  __constant double *cfl) {

  int       nelem, offset_b;

  if (get_group_id(0)+get_group_id(1)*get_num_groups(0) >= nblocks) return;

    // get sizes and shift pointers and direct-mapped data

    int blockId = blkmap[get_group_id(0) + get_group_id(1)*get_num_groups(0)  + block_offset];
    nelem    = nelems[blockId];
    offset_b = offset[blockId];
//
//  // copy indirect datasets into shared memory or zero increment
//

//
//  // process set elements
//

  int map0idx;
  int map1idx;
  int map2idx;
  int map3idx;
  const int n=get_local_id(0);
  
  if(n>=nelem) return;
  
    map0idx = ind_arg0_map_data[n + offset_b + set_size * 0];
    map1idx = ind_arg0_map_data[n + offset_b + set_size * 1];
    map2idx = ind_arg0_map_data[n + offset_b + set_size * 2];
    map3idx = ind_arg0_map_data[n + offset_b + set_size * 3];
      // user-supplied kernel call
      adt_calc(  &ind_arg0[2 * map0idx], /* x1  */
                 &ind_arg0[2 * map1idx], /* x2  */
                 &ind_arg0[2 * map2idx], /* x3  */
                 &ind_arg0[2 * map3idx], /* x4  */
                 &arg4[(n+offset_b)*4],  /* q   */
                 &arg5[(n+offset_b)*1],  /* adt */
                 *gam,
                 *gm1,
                 *cfl);

}

