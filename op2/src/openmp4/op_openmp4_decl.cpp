//
// This file implements the OP2 user-level functions for the OpenMP4.5 backend
//

#include <omp.h>

#include <op_lib_c.h>
#include <op_rt_support.h>

void cutilDeviceInit(int argc, char **argv);

void op_mvHostToDevice(void **map, int size);

void op_cpHostToDevice(void **data_d, void **data_h, int size);

void op_cuda_exit();

void op_cuda_get_data(op_dat dat);
//
// OpenMP-specific OP2 functions
//

void op_init_soa(int argc, char **argv, int diags, int soa) {
  OP_auto_soa = soa;
  op_init(argc, argv, diags);
}

void op_init(int argc, char **argv, int diags) {
  cutilDeviceInit(argc,argv);
  op_init_core(argc, argv, diags);
}

void op_mpi_init(int argc, char **argv, int diags, int global, int local) {
  cutilDeviceInit(argc,argv);
  op_init_core(argc, argv, diags);
}

void op_mpi_init_soa(int argc, char **argv, int diags, int global, int local,
                     int soa) {
  OP_auto_soa = soa;
  op_mpi_init(argc, argv, diags, global, local);
}

//_____________________________________________________________________________

op_dat op_decl_dat_char(op_set set, int dim, char const *type, int size,
                        char *data, char const *name) {
  op_dat dat = op_decl_dat_core(set, dim, type, size, data, name);

  // transpose data
  if (strstr(type, ":soa") != NULL || (OP_auto_soa && dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set->size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set->size + element_size * j + c] =
              data[dat->size * j + element_size * i + c];
        }
      }
    }
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(temp_data),
                      dat->size * set->size);
    free(temp_data);
  } else {
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data),
                      dat->size * set->size);
  }

  return dat;
}


op_dat op_decl_dat_temp_char(op_set set, int dim, char const *type, int size,
                             char const *name) {
  char *data = NULL;
  op_dat dat = op_decl_dat_temp_core(set, dim, type, size, data, name);

  for (size_t i = 0; i < set->size * dim * size; i++)
    dat->data[i] = 0;
  dat->user_managed = 0;

  // transpose data
  if (strstr(type, ":soa") != NULL || (OP_auto_soa && dim > 1)) {
    char *temp_data = (char *)malloc(dat->size * set->size * sizeof(char));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set->size + element_size * j + c] =
              data[dat->size * j + element_size * i + c];
        }
      }
    }
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(temp_data),
                      dat->size * set->size);
    free(temp_data);
  } else {
    op_cpHostToDevice((void **)&(dat->data_d), (void **)&(dat->data),
                      dat->size * set->size);
  }

  return dat;
}

int op_free_dat_temp_char(op_dat dat) {
  // free data on device
 #pragma omp target exit data map(delete:dat->data_d[:dat->size * dat->set->size])
  return op_free_dat_temp_core(dat);
}

op_set op_decl_set(int size, char const *name) {
  return op_decl_set_core(size, name);
}

op_map op_decl_map(op_set from, op_set to, int dim, int *imap,
                   char const *name) {
  op_map map = op_decl_map_core(from, to, dim, imap, name);

  int set_size = map->from->size;
  int *temp_map = (int *)malloc(map->dim * set_size * sizeof(int));
  for (int i = 0; i < map->dim; i++) {
    for (int j = 0; j < set_size; j++) {
      temp_map[i * set_size + j] = map->map[map->dim * j + i];
    }
  }

  op_cpHostToDevice((void **)&(map->map_d), (void **)&(temp_map),
                    map->dim * set_size * sizeof(int));
  free(temp_map);
  return map;
}

op_arg op_arg_dat(op_dat dat, int idx, op_map map, int dim, char const *type,
                  op_access acc) {
  return op_arg_dat_core(dat, idx, map, dim, type, acc);
}

op_arg op_opt_arg_dat(int opt, op_dat dat, int idx, op_map map, int dim,
                      char const *type, op_access acc) {
  return op_opt_arg_dat_core(opt, dat, idx, map, dim, type, acc);
}

op_arg op_arg_gbl_char(char *data, int dim, const char *type, int size,
                       op_access acc) {
  return op_arg_gbl_core(1, data, dim, type, size, acc);
}

op_arg op_opt_arg_gbl_char(int opt, char *data, int dim, const char *type,
                           int size, op_access acc) {
  return op_arg_gbl_core(opt, data, dim, type, size, acc);
}

int op_get_size(op_set set) { return set->size; }

void op_printf(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void op_print(const char *line) { printf("%s\n", line); }

void op_timers(double *cpu, double *et) { op_timers_core(cpu, et); }

int getSetSizeFromOpArg(op_arg *arg) {
  return arg->opt ? arg->dat->set->size : 0;
}

void op_renumber(op_map base) { (void)base; }

int getHybridGPU() { return OP_hybrid_gpu; }

void op_exit() {
  op_cuda_exit(); // frees dat_d memory
  op_rt_exit();   // frees plan memory
  op_exit_core(); // frees lib core variables
}

void op_timing_output() {
  op_timing_output_core();
  printf("Total plan time: %8.4f\n", OP_plan_time);
}

void op_timings_to_csv(const char *outputFileName) {
  FILE *outputFile = fopen(outputFileName, "w");
  if (outputFile == NULL) {
    printf("ERROR: Failed to open file for writing: '%s'\n", outputFileName);
  }
  else {
    fprintf(outputFile, "rank,thread,nranks,nthreads,count,total time,plan time,mpi time,GB used,GB total,kernel name\n");
  }

  if (outputFile != NULL) {
    for (int n = 0; n < OP_kern_max; n++) {
      if (OP_kernels[n].count > 0) {
        if (OP_kernels[n].ntimes == 1 && OP_kernels[n].times[0] == 0.0f &&
            OP_kernels[n].time != 0.0f) {
          // This library is being used by an OP2 translation made with the
          // older
          // translator with older timing logic. Adjust to new logic:
          OP_kernels[n].times[0] = OP_kernels[n].time;
        }

        for (int thr=0; thr<OP_kernels[n].ntimes; thr++) {
          double kern_time = OP_kernels[n].times[thr];
          if (kern_time == 0.0) {
            continue;
          }
          double plan_time = OP_kernels[n].plan_time;
          double mpi_time = OP_kernels[n].mpi_time;
          fprintf(outputFile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%s\n", 0, thr, 1,
                  OP_kernels[n].ntimes, OP_kernels[n].count, kern_time,
                  plan_time, mpi_time, OP_kernels[n].transfer / 1e9f,
                  OP_kernels[n].transfer2 / 1e9f, OP_kernels[n].name);
        }
      }
    }

    fclose(outputFile);
  }
}

void op_print_dat_to_binfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
    op_cuda_get_data(dat);
  op_print_dat_to_binfile_core(dat, file_name);
}

void op_print_dat_to_txtfile(op_dat dat, const char *file_name) {
  // need to get data from GPU
  op_cuda_get_data(dat);
  op_print_dat_to_txtfile_core(dat, file_name);
}

void op_upload_all() {
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    int set_size = dat->set->size;
    if (dat->data_d) {
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
        memcpy(dat->data_d,dat->data,set_size * dat->size * sizeof(char));
      }
      #pragma omp target update to(dat->data_d[:set_size * dat->size * sizeof(char)])
      dat->dirty_hd = 0;
    }
  }
}

void op_fetch_data_char(op_dat dat, char *usr_ptr) {
  op_cuda_get_data(dat);
  // need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)dat->data, dat->set->size * dat->size);
}

void op_fetch_data_idx_char(op_dat dat, char *usr_ptr, int low, int high) {
  op_cuda_get_data(dat);
  if (low < 0 || high > dat->set->size - 1) {
    printf("op_fetch_data: Indices not within range of elements held in %s\n",
           dat->name);
    exit(2);
  }
  // need to copy data into memory pointed to by usr_ptr
  memcpy((void *)usr_ptr, (void *)&dat->data[low * dat->size],
         (high + 1) * dat->size);
}

// Dummy for OpenMP compile

typedef struct {
} op_export_core;

typedef op_export_core *op_export_handle;

typedef struct {
} op_import_core;

typedef op_import_core *op_import_handle;

op_import_handle op_import_init_size(int nprocs, int *proclist, op_dat mark) {

  exit(1);
}

op_import_handle op_import_init(op_export_handle exp_handle, op_dat coords,
                                op_dat mark) {

  exit(1);
}

op_export_handle op_export_init(int nprocs, int *proclist, op_map cellsToNodes,
                                op_set sp_nodes, op_dat coords, op_dat mark) {

  exit(1);
}

void op_theta_init(op_export_handle handle, int *bc_id, double *dtheta_exp,
                   double *dtheta_imp, double *alpha) {

  exit(1);
}

void op_inc_theta(op_export_handle handle, int *bc_id, double *dtheta_exp,
                  double *dtheta_imp) {

  exit(1);
}

void op_export_data(op_export_handle handle, op_dat dat) { exit(1); }

void op_import_data(op_import_handle handle, op_dat dat) { exit(1); }
