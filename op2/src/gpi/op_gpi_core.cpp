#include <op_perf_common.h>
#include <op_util.h>
#include <op_lib_core.h>

#include <GASPI.h>

// Timing
double t1, t2, c1, c2;

void op_gpi_reduce_combined(op_arg *args, int nargs){
  op_timers_core(&c1, &t1);
  int nreductions = 0;

  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ)
      nreductions++;
  }

  op_arg *arg_list = (op_arg *)xmalloc(nreductions * sizeof(op_arg));
  nreductions = 0;
  int nbytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ) {
      arg_list[nreductions++] = args[i];
      nbytes += args[i].size;
    }
  }
  char *data = (char *)xmalloc(nbytes * sizeof(char)); // TODO
  /*
  char *data;
  gaspi_segment_id_t send_data_segment_id = ?;
  GPI_SAFE( gaspi_segment_ptr(send_data_segment_id, &data);
  // don't know what's going on with the segments
  */

  int char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    for (int j = 0; j < arg_list[i].size; j++)
      data[char_counter++] = arg_list[i].data[j];
  }

  int comm_size, comm_rank;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);  // TODO
  MPI_Comm_rank(OP_MPI_WORLD, &comm_rank);  // TODO
  char *result = (char *)xmalloc(comm_size * nbytes * sizeof(char));
  MPI_Allgather(data, nbytes, MPI_CHAR, result, nbytes, MPI_CHAR, OP_MPI_WORLD); // TODO
  /*
  gaspi_segment_id_t local_segment = ?;
  gaspi_offset_t local_offset = 0;
  gaspi_size_t size = ?;
  gaspi_segment_id_t remote_segment = local_segment;
  gaspi_offset_t remote_offset = size;
  gaspi_queue_id_t queue = 0;
  gaspi_group_t group = GASPI_GROUP_ALL;
  gaspi_timeout_t = GASPI_BLOCK;
  
  GPI_allgather(
    local_segment, 
    local_offset, 
    size,
    remote_segment,
    remote_offset, 
    queue, 
    group, 
    timeout 
  );
  
  char *result;
  GPI_SAFE( gaspi_segment_ptr(local_segment, &result);
  // may need to copy data into new result array instead
  */
  char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    if (strcmp(arg_list[i].type, "double") == 0 ||
        strcmp(arg_list[i].type, "r8") == 0) {
      double *output = (double *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] +=
                  ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((double *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((double *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((double *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "float") == 0 ||
        strcmp(arg_list[i].type, "r4") == 0 ||
        strcmp(arg_list[i].type, "real*4") == 0) {
      float *output = (float *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] +=
                  ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((float *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((float *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((float *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "int") == 0 ||
        strcmp(arg_list[i].type, "i4") == 0 ||
        strcmp(arg_list[i].type, "integer*4") == 0) {
      int *output = (int *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] += ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((int *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((int *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((int *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    if (strcmp(arg_list[i].type, "bool") == 0 ||
        strcmp(arg_list[i].type, "logical") == 0) {
      bool *output = (bool *)arg_list[i].data;
      for (int rank = 0; rank < comm_size; rank++) {
        if (rank != comm_rank) {
          if (arg_list[i].acc == OP_INC) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] += ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MIN) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] <
                          ((bool *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_MAX) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] >
                          ((bool *)(result + char_counter + nbytes * rank))[j]
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          } else if (arg_list[i].acc == OP_WRITE) {
            for (int j = 0; j < arg_list[i].dim; j++) {
              output[j] =
                  output[j] != 0.0
                      ? output[j]
                      : ((bool *)(result + char_counter + nbytes * rank))[j];
            }
          }
        }
      }
    }
    char_counter += arg_list[i].size;
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
  op_free(arg_list);  
  op_free(data);  // TODO
  op_free(result);  // TODO
}

void op_gpi_reduce_float(op_arg *arg, float *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.
  if (arg->data == NULL)
    return;
  (void)data;
  op_timers_core(&c1, &t1);
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    float result_static;
    float *result;
    if (arg->dim > 1 && arg->acc != OP_WRITE)
      result = (float *)calloc(arg->dim, sizeof(float));
    else
      result = &result_static;

    /*
    copy data into misc buffer
    float* sendData;
    gaspi_segment_id_t send_data_segment_id = ?;
    GPI_SAFE( gaspi_segment_ptr(send_data_segment_id, sendData);
    memcpy(sendData, arg->data, arg->dim * sizeof(float))

    set receive section to rest of buffer
    float* receive = sendData + arg->dim*sizeof(float);

    gaspi_

    */
    // entire if statement can be removed in GPI I think
    if (arg->acc == OP_INC) // global reduction
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_SUM,
                    OP_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float) * arg->dim);
    } else if (arg->acc == OP_MAX) // global maximum
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MAX,
                    OP_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float) * arg->dim);
      ;
    } else if (arg->acc == OP_MIN) // global minimum
    {
      MPI_Allreduce((float *)arg->data, result, arg->dim, MPI_FLOAT, MPI_MIN,
                    OP_MPI_WORLD);
      memcpy(arg->data, result, sizeof(float) * arg->dim);
    } else if (arg->acc == OP_WRITE) // any
    {
      int size;
      MPI_Comm_size(OP_MPI_WORLD, &size);
      result = (float *)calloc(arg->dim * size, sizeof(float));
      MPI_Allgather((float *)arg->data, arg->dim, MPI_FLOAT, result, arg->dim,
                    MPI_FLOAT, OP_MPI_WORLD);
      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i * arg->dim + j] != 0.0f)
            result[j] = result[i * arg->dim + j];
        }
      }
      memcpy(arg->data, result, sizeof(float) * arg->dim);
      if (arg->dim == 1)
        op_free(result);
    }
    if (arg->dim > 1)
      op_free(result);
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
}

void op_gpi_reduce_double(op_arg *arg, double *data){

}

void op_gpi_reduce_int(op_arg *arg, int *data){

}

void op_gpi_reduce_bool(op_arg *arg, bool *data){

}