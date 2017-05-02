
//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <omp.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

//
// routines to move arrays to/from GPU device
//

void op_mvHostToDevice(void **map, int size) {
  if (!OP_hybrid_gpu)
    return;
  char *temp = (char*)*map;
  #pragma omp target enter data map(to: temp[:size])
  #pragma omp target update to(temp[:size])
  //TODO test
}

void op_cpHostToDevice(void **data_d, void **data_h, int size) {
  if (!OP_hybrid_gpu)
    return;
  *data_d = (char*)op_malloc(size);
  memcpy(*data_d, *data_h, size);
  char *tmp = (char *)*data_d;
  //TODO jo igy? decl miatt kell az enter data elm.
  #pragma omp target enter data map(to: tmp[:size])
  #pragma omp target update to(tmp[:size])
}

op_plan *op_plan_get(char const *name, op_set set, int part_size, int nargs,
                     op_arg *args, int ninds, int *inds) {
  return op_plan_get_stage(name, set, part_size, nargs, args, ninds, inds,
                           OP_STAGE_ALL);
}

op_plan *op_plan_get_stage(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging) {
  op_plan *plan =
      op_plan_core(name, set, part_size, nargs, args, ninds, inds, staging);
  if (!OP_hybrid_gpu)
    return plan;

  int set_size = set->size;
  for (int i = 0; i < nargs; i++) {
    if (args[i].idx != -1 && args[i].acc != OP_READ) {
      set_size += set->exec_size;
      break;
    }
  }

  if (plan->count == 1) {
    int *offsets = (int *)malloc((plan->ninds_staged + 1) * sizeof(int));
    offsets[0] = 0;
    for (int m = 0; m < plan->ninds_staged; m++) {
      int count = 0;
      for (int m2 = 0; m2 < nargs; m2++)
        if (plan->inds_staged[m2] == m)
          count++;
      offsets[m + 1] = offsets[m] + count;
    }
    op_mvHostToDevice((void **)&(plan->ind_map),
                      offsets[plan->ninds_staged] * set_size * sizeof(int));
    for (int m = 0; m < plan->ninds_staged; m++) {
      plan->ind_maps[m] = &plan->ind_map[set_size * offsets[m]];
    }
    free(offsets);

    int counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL)
        counter++;
    op_mvHostToDevice((void **)&(plan->loc_map),
                      sizeof(short) * counter * set_size);
    counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL) {
        plan->loc_maps[m] = &plan->loc_map[set_size * counter];
        counter++;
      }

    op_mvHostToDevice((void **)&(plan->ind_sizes),
                      sizeof(int) * plan->nblocks * plan->ninds_staged);
    op_mvHostToDevice((void **)&(plan->ind_offs),
                      sizeof(int) * plan->nblocks * plan->ninds_staged);
    op_mvHostToDevice((void **)&(plan->nthrcol), sizeof(int) * plan->nblocks);
    op_mvHostToDevice((void **)&(plan->thrcol), sizeof(int) * set_size);
    op_mvHostToDevice((void **)&(plan->col_reord), sizeof(int) * set_size);
    op_mvHostToDevice((void **)&(plan->offset), sizeof(int) * plan->nblocks);
    plan->offset_d = plan->offset;
    op_mvHostToDevice((void **)&(plan->nelems), sizeof(int) * plan->nblocks);
    plan->nelems_d = plan->nelems;
    op_mvHostToDevice((void **)&(plan->blkmap), sizeof(int) * plan->nblocks);
    plan->blkmap_d = plan->blkmap;
  }

  return plan;
}

void op_cuda_exit() {
  if (!OP_hybrid_gpu)
    return;
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    #pragma omp target exit data map(from: (item->dat)->data_d)
    free((item->dat)->data_d);
  }
  /*
  for (int ip = 0; ip < OP_plan_index; ip++) {
    OP_plans[ip].ind_map = NULL;
    OP_plans[ip].loc_map = NULL;
    OP_plans[ip].ind_sizes = NULL;
    OP_plans[ip].ind_offs = NULL;
    OP_plans[ip].nthrcol = NULL;
    OP_plans[ip].thrcol = NULL;
    OP_plans[ip].col_reord = NULL;
    OP_plans[ip].offset = NULL;
    OP_plans[ip].nelems = NULL;
    OP_plans[ip].blkmap = NULL;
  }
  */
  // cudaDeviceReset ( );
}


//
// routines to resize constant/reduct arrays, if necessary
//

void reallocConstArrays(int consts_bytes) {
  (void) consts_bytes;
}

void reallocReductArrays(int reduct_bytes) {
  (void) reduct_bytes;
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice(int consts_bytes) {  
  (void) consts_bytes;
}

void mvReductArraysToDevice(int reduct_bytes) {
  (void) reduct_bytes;
}

void mvReductArraysToHost(int reduct_bytes) {
  (void) reduct_bytes;
}

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void op_cuda_get_data(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return; 
  
  #pragma omp target update from(dat->data_d[:dat->size * dat->set->size])
  
  // transpose data
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
  
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < dat->set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              dat->data_d[element_size * i * dat->set->size + element_size * j +
                        c];
        }
      }
    }
  } else {
    memcpy(dat->data,dat->data_d,dat->size * dat->set->size); 
  }

}

void deviceSync() {
//  cutilSafeCall(cudaDeviceSynchronize());
}
#ifndef OPMPI

void cutilDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  // copy one scalar to initialize OpenMP env. 
  // Improvement: later we can set default device.
  int tmp=0;
  #pragma omp target enter data map(to:tmp)

  OP_hybrid_gpu = 1;
}


void op_upload_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  int set_size = dat->set->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data_d[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
  } else {
    memcpy(dat->data_d,dat->data,dat->size * dat->set->size); 
  }
  #pragma omp target update to(dat->data_d[:set_size*dat->size])
}

void op_download_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  #pragma omp target update from(dat->data_d[:dat->size * dat->set->size])
  int set_size = dat->set->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              dat->data_d[element_size * i * set_size + element_size * j + c];
        }
      }
    }
  } else {
    memcpy(dat->data,dat->data_d,dat->size * dat->set->size); 
  }
}

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) { //TODO itt a download + dirty allitas ekv egy getdata hivassal
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 2) {
      op_download_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  return set->size;
}

void op_mpi_set_dirtybit(int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].opt == 1) && (args[n].argtype == OP_ARG_DAT) &&
        (args[n].acc == OP_INC || args[n].acc == OP_WRITE ||
         args[n].acc == OP_RW)) {
      args[n].dat->dirty_hd = 1;
    }
  }
}

int op_mpi_halo_exchanges_cuda(op_set set, int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 1) {
      op_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  return set->size;
}

void op_mpi_set_dirtybit_cuda(int nargs, op_arg *args) {
  for (int n = 0; n < nargs; n++) {
    if ((args[n].opt == 1) && (args[n].argtype == OP_ARG_DAT) &&
        (args[n].acc == OP_INC || args[n].acc == OP_WRITE ||
         args[n].acc == OP_RW)) {
      args[n].dat->dirty_hd = 2;
    }
  }
}

void op_mpi_wait_all(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_wait_all_cuda(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_reset_halos(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
}

void op_mpi_barrier() {}

void *op_mpi_perf_time(const char *name, double time) {
  (void)name;
  (void)time;
  return (void *)name;
}

#ifdef COMM_PERF
void op_mpi_perf_comms(void *k_i, int nargs, op_arg *args) {
  (void)k_i;
  (void)nargs;
  (void)args;
}
#endif

void op_mpi_reduce_float(op_arg *args, float *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_double(op_arg *args, double *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_int(op_arg *args, int *data) {
  (void)args;
  (void)data;
}

void op_mpi_reduce_bool(op_arg *args, bool *data) {
  (void)args;
  (void)data;
}

void op_partition(const char *lib_name, const char *lib_routine,
                  op_set prime_set, op_map prime_map, op_dat coords) {
  (void)lib_name;
  (void)lib_routine;
  (void)prime_set;
  (void)prime_map;
  (void)coords;
}

void op_partition_reverse() {}

void op_compute_moment(double t, double *first, double *second) {
  *first = t;
  *second = t * t;
}

int op_is_root() { return 1; }
#endif
