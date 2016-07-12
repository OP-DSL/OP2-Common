//
// auto-generated by op2.py
//

// header
#ifdef GPUPASS
#define op_par_loop_save_soln op_par_loop_save_soln_gpu
#define op_par_loop_adt_calc op_par_loop_adt_calc_gpu
#define op_par_loop_res_calc op_par_loop_res_calc_gpu
#define op_par_loop_bres_calc op_par_loop_bres_calc_gpu
#define op_par_loop_update op_par_loop_update_gpu
#include "airfoil_kernels.cu"
#undef op_par_loop_save_soln
#undef op_par_loop_adt_calc
#undef op_par_loop_res_calc
#undef op_par_loop_bres_calc
#undef op_par_loop_update
#else
#define op_par_loop_save_soln op_par_loop_save_soln_cpu
#define op_par_loop_adt_calc op_par_loop_adt_calc_cpu
#define op_par_loop_res_calc op_par_loop_res_calc_cpu
#define op_par_loop_bres_calc op_par_loop_bres_calc_cpu
#define op_par_loop_update op_par_loop_update_cpu
#include "airfoil_kernels.cpp"
#undef op_par_loop_save_soln
#undef op_par_loop_adt_calc
#undef op_par_loop_res_calc
#undef op_par_loop_bres_calc
#undef op_par_loop_update

// user kernel files

void op_par_loop_save_soln_gpu(char const *name, op_set set, op_arg arg0,
                               op_arg arg1);

// GPU host stub function
#if OP_HYBRID_GPU
void op_par_loop_save_soln(char const *name, op_set set, op_arg arg0,
                           op_arg arg1) {

  if (OP_hybrid_gpu) {
    op_par_loop_save_soln_gpu(name, set, arg0, arg1);

  } else {
    op_par_loop_save_soln_cpu(name, set, arg0, arg1);
  }
}
#else
void op_par_loop_save_soln(char const *name, op_set set, op_arg arg0,
                           op_arg arg1) {

  op_par_loop_save_soln_gpu(name, set, arg0, arg1);
}
#endif // OP_HYBRID_GPU

void op_par_loop_adt_calc_gpu(char const *name, op_set set, op_arg arg0,
                              op_arg arg1, op_arg arg2, op_arg arg3,
                              op_arg arg4, op_arg arg5);

// GPU host stub function
#if OP_HYBRID_GPU
void op_par_loop_adt_calc(char const *name, op_set set, op_arg arg0,
                          op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                          op_arg arg5) {

  if (OP_hybrid_gpu) {
    op_par_loop_adt_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);

  } else {
    op_par_loop_adt_calc_cpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);
  }
}
#else
void op_par_loop_adt_calc(char const *name, op_set set, op_arg arg0,
                          op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                          op_arg arg5) {

  op_par_loop_adt_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);
}
#endif // OP_HYBRID_GPU

void op_par_loop_res_calc_gpu(char const *name, op_set set, op_arg arg0,
                              op_arg arg1, op_arg arg2, op_arg arg3,
                              op_arg arg4, op_arg arg5, op_arg arg6,
                              op_arg arg7);

// GPU host stub function
#if OP_HYBRID_GPU
void op_par_loop_res_calc(char const *name, op_set set, op_arg arg0,
                          op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                          op_arg arg5, op_arg arg6, op_arg arg7) {

  if (OP_hybrid_gpu) {
    op_par_loop_res_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5,
                             arg6, arg7);

  } else {
    op_par_loop_res_calc_cpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5,
                             arg6, arg7);
  }
}
#else
void op_par_loop_res_calc(char const *name, op_set set, op_arg arg0,
                          op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                          op_arg arg5, op_arg arg6, op_arg arg7) {

  op_par_loop_res_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                           arg7);
}
#endif // OP_HYBRID_GPU

void op_par_loop_bres_calc_gpu(char const *name, op_set set, op_arg arg0,
                               op_arg arg1, op_arg arg2, op_arg arg3,
                               op_arg arg4, op_arg arg5);

// GPU host stub function
#if OP_HYBRID_GPU
void op_par_loop_bres_calc(char const *name, op_set set, op_arg arg0,
                           op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                           op_arg arg5) {

  if (OP_hybrid_gpu) {
    op_par_loop_bres_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);

  } else {
    op_par_loop_bres_calc_cpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);
  }
}
#else
void op_par_loop_bres_calc(char const *name, op_set set, op_arg arg0,
                           op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4,
                           op_arg arg5) {

  op_par_loop_bres_calc_gpu(name, set, arg0, arg1, arg2, arg3, arg4, arg5);
}
#endif // OP_HYBRID_GPU

void op_par_loop_update_gpu(char const *name, op_set set, op_arg arg0,
                            op_arg arg1, op_arg arg2, op_arg arg3, op_arg arg4);

// GPU host stub function
#if OP_HYBRID_GPU
void op_par_loop_update(char const *name, op_set set, op_arg arg0, op_arg arg1,
                        op_arg arg2, op_arg arg3, op_arg arg4) {

  if (OP_hybrid_gpu) {
    op_par_loop_update_gpu(name, set, arg0, arg1, arg2, arg3, arg4);

  } else {
    op_par_loop_update_cpu(name, set, arg0, arg1, arg2, arg3, arg4);
  }
}
#else
void op_par_loop_update(char const *name, op_set set, op_arg arg0, op_arg arg1,
                        op_arg arg2, op_arg arg3, op_arg arg4) {

  op_par_loop_update_gpu(name, set, arg0, arg1, arg2, arg3, arg4);
}
#endif // OP_HYBRID_GPU
#endif
