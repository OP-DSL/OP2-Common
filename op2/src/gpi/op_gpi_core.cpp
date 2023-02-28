#include <op_perf_common.h>
#include <op_util.h>
#include <op_lib_core.h>

#include <op_gpi_core.h>
#include <gpi_utils.h>
#include <GPI_gather.h>

#include <GASPI.h>

//#define MSC_SEGMENT_ID 1
//#define OP_GPI_WORLD GASPI_GROUP_ALL
//#define GPI_TIMOUT GASPI_BLOCK

// Timing
double t1, t2, c1, c2;
extern op_kernel* OP_kernels;
extern int OP_kern_max, OP_kern_curr;


void op_gpi_reduce_combined(op_arg *args, int nargs){
  //printf("start\n");
  int nreductions = 0;
  
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ)
      nreductions++;
  }
  
  op_arg *arg_list = (op_arg *)malloc(nreductions * sizeof(op_arg));
  nreductions = 0;
  int nbytes = 0;
  for (int i = 0; i < nargs; i++) {
    if (args[i].argtype == OP_ARG_GBL && args[i].acc != OP_READ) {
      arg_list[nreductions++] = args[i];
      nbytes += args[i].size;
    }
  }
  //char *data = (char *)xmalloc(nbytes * sizeof(char)); // TODO
  
  char *data;
  gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
  SUCCESS_OR_DIE( gaspi_segment_ptr(send_data_segment_id, &data) );

  int char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    for (int j = 0; j < arg_list[i].size; j++)
      data[char_counter++] = arg_list[i].data[j];
  }

  /*
  int comm_size, comm_rank;
  MPI_Comm_size(OP_MPI_WORLD, &comm_size);  // TODO
  MPI_Comm_rank(OP_MPI_WORLD, &comm_rank);  // TODO
  char *result = (char *)xmalloc(comm_size * nbytes * sizeof(char));
  MPI_Allgather(data, nbytes, MPI_CHAR, result, nbytes, MPI_CHAR, OP_MPI_WORLD); // TODO
  */

  
  int comm_size, comm_rank;

	SUCCESS_OR_DIE( gaspi_proc_rank(&comm_rank) );
  SUCCESS_OR_DIE( gaspi_group_size(GASPI_GROUP_ALL,&comm_size) );
  
  gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
  gaspi_offset_t local_offset = 0;
  gaspi_size_t size = nbytes;
  gaspi_segment_id_t remote_segment = local_segment;
  gaspi_offset_t remote_offset = size;
  gaspi_queue_id_t queue = 1;
  gaspi_group_t group = GASPI_GROUP_ALL;
  gaspi_timeout_t timeout = GASPI_BLOCK;
  
  
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
  //printf("hello\n");
  //fflush(stdout);

  char *result;
  SUCCESS_OR_DIE( gaspi_segment_ptr(local_segment, &result) );
  result += nbytes;
  //printf("Rank %d storing %d floats: ", comm_rank, comm_size*nbytes/sizeof(float));
  for (int i=0; i<comm_size*nbytes/sizeof(float); i++){
    //printf("%f ", result[i]);
  }
  //printf("\n");
  //fflush(stdout);

  
  // may need to copy data into new result array instead
  char_counter = 0;
  for (int i = 0; i < nreductions; i++) {
    //printf("type %d", strcmp(arg_list[i].type, "float"));
    //fflush(stdout);
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
      //printf("in float %d\n", comm_size);
      //fflush(stdout);
      for (int rank = 0; rank < comm_size; rank++) {
        //printf("arg, %d \n", arg_list[i].acc);
        //fflush(stdout);
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
  
  free(arg_list);
}


void op_gpi_reduce_float(op_arg *arg, float *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	SUCCESS_OR_DIE( gaspi_proc_rank(&comm_rank) );
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    float* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    SUCCESS_OR_DIE( gaspi_segment_ptr(send_data_segment_id, &send_data) );
    memcpy(send_data, arg->data, arg->dim * sizeof(float));

    // set receive section to rest of buffer
    float* receive = send_data + arg->dim*sizeof(float);

    int datasize = sizeof(float) * arg->dim;

    gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
    gaspi_offset_t local_offset = 0;
    gaspi_size_t size = (gaspi_size_t) datasize;
    gaspi_segment_id_t remote_segment = local_segment;
    gaspi_offset_t remote_offset = size;
    gaspi_queue_id_t queue = 1;  // TODO
    gaspi_group_t group = OP_GPI_WORLD;
    gaspi_timeout_t timeout = GPI_TIMOUT;
    
    if (arg->acc == OP_WRITE){
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

      float* result = receive;

      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i * arg->dim + j] != 0.0f)
            // flatten one-hot encoded data
            // not sure why we can assume that
            result[j] = result[i * arg->dim + j];
        }
      }

      memcpy(arg->data, result, sizeof(float) * arg->dim);
    }else{
      gaspi_operation_t operation;
      if (arg->acc == OP_INC){
        operation = GASPI_OP_SUM;
      } else if (arg->acc == OP_MIN){
        operation = GASPI_OP_MIN;
      } else if (arg->acc == OP_MAX){
        operation = GASPI_OP_MAX;
      }

      SUCCESS_OR_DIE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_FLOAT,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMOUT// gaspi_timeout_t timeout )
      ) );

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_double(op_arg *arg, double *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	SUCCESS_OR_DIE( gaspi_proc_rank(&comm_rank) );
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    double* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    SUCCESS_OR_DIE( gaspi_segment_ptr(send_data_segment_id, &send_data) );
    memcpy(send_data, arg->data, arg->dim * sizeof(double));

    // set receive section to rest of buffer
    double* receive = send_data + arg->dim*sizeof(double);

    int datasize = sizeof(double) * arg->dim;

    gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
    gaspi_offset_t local_offset = 0;
    gaspi_size_t size = (gaspi_size_t) datasize;
    gaspi_segment_id_t remote_segment = local_segment;
    gaspi_offset_t remote_offset = size;
    gaspi_queue_id_t queue = 1;  // TODO
    gaspi_group_t group = OP_GPI_WORLD;
    gaspi_timeout_t timeout = GPI_TIMOUT;
    
    if (arg->acc == OP_WRITE){
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

      double* result = receive;

      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i * arg->dim + j] != 0.0f)
            // flatten one-hot encoded data
            // not sure why we can assume that
            result[j] = result[i * arg->dim + j];
        }
      }

      memcpy(arg->data, result, sizeof(double) * arg->dim);
    }else{
      gaspi_operation_t operation;
      if (arg->acc == OP_INC){
        operation = GASPI_OP_SUM;
      } else if (arg->acc == OP_MIN){
        operation = GASPI_OP_MIN;
      } else if (arg->acc == OP_MAX){
        operation = GASPI_OP_MAX;
      }

      SUCCESS_OR_DIE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_DOUBLE,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMOUT// gaspi_timeout_t timeout )
      ) );

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_int(op_arg *arg, int *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	SUCCESS_OR_DIE( gaspi_proc_rank(&comm_rank) );
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    int* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    SUCCESS_OR_DIE( gaspi_segment_ptr(send_data_segment_id, &send_data) );
    memcpy(send_data, arg->data, arg->dim * sizeof(int));

    // set receive section to rest of buffer
    int* receive = send_data + arg->dim*sizeof(int);

    int datasize = sizeof(int) * arg->dim;

    gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
    gaspi_offset_t local_offset = 0;
    gaspi_size_t size = (gaspi_size_t) datasize;
    gaspi_segment_id_t remote_segment = local_segment;
    gaspi_offset_t remote_offset = size;
    gaspi_queue_id_t queue = 1;  // TODO
    gaspi_group_t group = OP_GPI_WORLD;
    gaspi_timeout_t timeout = GPI_TIMOUT;
    
    if (arg->acc == OP_WRITE){
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

      int* result = receive;

      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i * arg->dim + j] != 0.0f)
            // flatten one-hot encoded data
            // not sure why we can assume that
            result[j] = result[i * arg->dim + j];
        }
      }

      memcpy(arg->data, result, sizeof(int) * arg->dim);
    }else{
      gaspi_operation_t operation;
      if (arg->acc == OP_INC){
        operation = GASPI_OP_SUM;
      } else if (arg->acc == OP_MIN){
        operation = GASPI_OP_MIN;
      } else if (arg->acc == OP_MAX){
        operation = GASPI_OP_MAX;
      }

      SUCCESS_OR_DIE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_INT,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMOUT// gaspi_timeout_t timeout )
      ) );

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_bool(op_arg *arg, bool *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	SUCCESS_OR_DIE( gaspi_proc_rank(&comm_rank) );
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    bool* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    SUCCESS_OR_DIE( gaspi_segment_ptr(send_data_segment_id, &send_data) );
    memcpy(send_data, arg->data, arg->dim * sizeof(bool));

    // set receive section to rest of buffer
    bool* receive = send_data + arg->dim*sizeof(bool);

    int datasize = sizeof(bool) * arg->dim;

    gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
    gaspi_offset_t local_offset = 0;
    gaspi_size_t size = (gaspi_size_t) datasize;
    gaspi_segment_id_t remote_segment = local_segment;
    gaspi_offset_t remote_offset = size;
    gaspi_queue_id_t queue = 1;  // TODO
    gaspi_group_t group = OP_GPI_WORLD;
    gaspi_timeout_t timeout = GPI_TIMOUT;
    
    if (arg->acc == OP_WRITE){
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

      bool* result = receive;

      for (int i = 1; i < size; i++) {
        for (int j = 0; j < arg->dim; j++) {
          if (result[i * arg->dim + j] != 0.0f)
            // flatten one-hot encoded data
            // not sure why we can assume that
            result[j] = result[i * arg->dim + j];
        }
      }

      memcpy(arg->data, result, sizeof(bool) * arg->dim);
    }else{
      // gaspi doesn't have a built in BOOL type, 
      // reduce_combined doesn't use gaspi_reduce,
      // so just use that
      op_gpi_reduce_combined(arg, 1);
    }
  }
}