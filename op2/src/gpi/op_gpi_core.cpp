
#include "gpi_utils.h"

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>
#include <op_lib_core.h>

#include <op_gpi_core.h>


#include "gpi_utils.h"

#include <GASPI.h>
/* PROBLEM: op_mpi_core stores all of the global variables that are all of the halo types and dat list.
 * Ideally would be best to move that around a bit so it's common rather than just MPI 
 */


// Timing
double t1, t2, c1, c2;
extern op_kernel* OP_kernels;
extern int OP_kern_max, OP_kern_curr;


/* Segment pointers */
char *eeh_segment_ptr;
char *ieh_segment_ptr;
char *enh_segment_ptr;
char *inh_segment_ptr;

char *msc_segment_ptr;


/* IS_COMMON 
 * Bascially a near perfect copy of op_mpi_halo_exchanges, only 2/3 lines changed. */
int op_gpi_halo_exchanges(op_set set, int nargs, op_arg *args){
  int size = set->size;
  int direct_flag = 1;

  if (OP_diags > 0) {
    int dummy;
    for (int n = 0; n < nargs; n++)
      op_arg_check(set, n, args[n], &dummy, "halo_exchange mpi");
  }

  if (OP_hybrid_gpu) {
    for (int n = 0; n < nargs; n++)
      if (args[n].opt && args[n].argtype == OP_ARG_DAT &&
          args[n].dat->dirty_hd == 2) {
        op_download_dat(args[n].dat);
        args[n].dat->dirty_hd = 0;
      }
  }

  // check if this is a direct loop
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;

  if (direct_flag == 1)
    return size;

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      size = set->size + set->exec_size;
      exec_flag = 1;
    }
  }
  op_timers_core(&c1, &t1);
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT) {
      if (args[n].map == OP_ID) {
        op_gpi_exchange_halo(&args[n], exec_flag);
      } else {
        // Check if dat-map combination was already done or if there is a
        // mismatch (same dat, diff map)
        int found = 0;
        int fallback = 0;
        for (int m = 0; m < nargs; m++) {
          if (m < n && args[n].dat == args[m].dat && args[n].map == args[m].map)
            found = 1;
          else if (args[n].dat == args[m].dat && args[n].map != args[m].map)
            fallback = 1;
        }
        // If there was a map mismatch with other argument, do full halo
        // exchange
        if (fallback)
          op_gpi_exchange_halo(&args[n], exec_flag);
        else if (!found) { // Otherwise, if partial halo exchange is enabled for
                           // this map, do it
          if (OP_map_partial_exchange[args[n].map->index])
            op_gpi_exchange_halo_partial(&args[n], exec_flag);
          else
            op_gpi_exchange_halo(&args[n], exec_flag);
        }
      }
    }
  }
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].gpi_time += t2 - t1;
  return size;
}

/* Wait for all args.
 * IS_COMMON
 * GPI replacement for op_mpi_wait_all.
 * MPI has a bad naming convention for this stuff, thus tried to clarify with _args */
void op_gpi_waitall_args(int nargs, op_arg *args){
    op_timers_core(&c1, &t1);
    for (int n = 0; n < nargs; n++) {
        op_gpi_waitall(&args[n]);
    }
    op_timers_core(&c2, &t2);
    if (OP_kern_max > 0)
        OP_kernels[OP_kern_curr].gpi_time += t2 - t1;
}

void *op_gpi_perf_time(const char *name, double time){
  return (void*)NULL;
}



//#define MSC_SEGMENT_ID 1
//#define OP_GPI_WORLD GASPI_GROUP_ALL
//#define GPI_TIMOUT GASPI_BLOCK



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
  GPI_SAFE( gaspi_segment_ptr(send_data_segment_id,(gaspi_pointer_t*) &data) ) 

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

	GPI_SAFE( gaspi_proc_rank((gaspi_rank_t*)&comm_rank) )
  GPI_SAFE( gaspi_group_size(GASPI_GROUP_ALL,(gaspi_number_t*)&comm_size) )
  
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
  GPI_SAFE( gaspi_segment_ptr(local_segment, (gaspi_pointer_t*)&result) );
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

	GPI_SAFE( gaspi_proc_rank((gaspi_rank_t*)&comm_rank) )
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    float* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    GPI_SAFE( gaspi_segment_ptr(send_data_segment_id, (gaspi_pointer_t*)&send_data) )
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
    gaspi_timeout_t timeout = GPI_TIMEOUT;
    
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

      GPI_SAFE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_FLOAT,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMEOUT// gaspi_timeout_t timeout )
      ) );

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_double(op_arg *arg, double *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	GPI_SAFE( gaspi_proc_rank((gaspi_rank_t*)&comm_rank) )
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    double* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    GPI_SAFE( gaspi_segment_ptr(send_data_segment_id, (gaspi_pointer_t*) &send_data) )
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
    gaspi_timeout_t timeout = GPI_TIMEOUT;
    
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

      GPI_SAFE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_DOUBLE,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMEOUT// gaspi_timeout_t timeout )
      ) )

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_int(op_arg *arg, int *data){
  // I have no idea what this data argument is?
  // it doesn't appear to be used at all.  
  int comm_size, comm_rank;

	GPI_SAFE( gaspi_proc_rank((gaspi_rank_t*)&comm_rank) );
 
  if (arg->data == NULL)
    return;
  (void)data;
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    int* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    GPI_SAFE( gaspi_segment_ptr(send_data_segment_id,(gaspi_pointer_t*) &send_data) );
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
    gaspi_timeout_t timeout = GPI_TIMEOUT;
    
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

      GPI_SAFE( gaspi_allreduce( 
        send_data,// gaspi_const_pointer_t buffer_send,
        (gaspi_pointer_t) receive,// gaspi_pointer_t buffer_receive,
        (gaspi_number_t) arg->dim,// gaspi_number_t num,
        operation,// gaspi_operation_t operation,
        GASPI_TYPE_INT,// gaspi_datatype_t datatype,
        GASPI_GROUP_ALL,// gaspi_group_t group,
        GPI_TIMEOUT// gaspi_timeout_t timeout )
      ) )

      memcpy(arg->data, receive, size);
    }
  }
}

void op_gpi_reduce_bool(op_arg *arg, bool *data){
  int comm_size, comm_rank;

	GPI_SAFE( gaspi_proc_rank((gaspi_rank_t*)&comm_rank) )
 
  if (arg->data == NULL)
    return;
  
  (void)data; /* Unused - avoids compiler warning */
  
  if (arg->argtype == OP_ARG_GBL && arg->acc != OP_READ) {
    // copy data into misc buffer
    bool* send_data;
    gaspi_segment_id_t send_data_segment_id = MSC_SEGMENT_ID;
    GPI_SAFE( gaspi_segment_ptr(send_data_segment_id, (gaspi_pointer_t*)&send_data) )
    memcpy(send_data, arg->data, arg->dim * sizeof(bool));

    // set receive section to rest of buffer
    bool* receive = send_data + arg->dim*sizeof(bool);

    int datasize = sizeof(bool) * arg->dim;

    gaspi_segment_id_t local_segment = MSC_SEGMENT_ID;
    gaspi_offset_t local_offset = 0;
    gaspi_size_t size = (gaspi_size_t) datasize;
    gaspi_segment_id_t remote_segment = local_segment;
    gaspi_offset_t remote_offset = size;
    gaspi_queue_id_t queue = OP2_GPI_QUEUE_ID;  // TODO
    gaspi_group_t group = OP_GPI_WORLD;
    gaspi_timeout_t timeout = GPI_TIMEOUT;
    
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


/* Best effort reproduction of MPI_Allgather.
 * Adds aditional assumption that the sizes exchanged by each process are identical.
 * Remote segment is implicitly your local receive segment by symmetry.*/
int GPI_allgather(gaspi_segment_id_t segment_id_local, /* Send segment */
                  gaspi_offset_t offset_local, /* address of data to be sent*/
                  gaspi_size_t size, /* size of sendcount * sizeof(sendtype). Assumes send elems= recv elems */
                  gaspi_segment_id_t segment_id_remote, /* Recv segment */
                  gaspi_offset_t offset_remote, /* base offset - incase segment has multiple purposes */
                  gaspi_queue_id_t queue, /* queue id for write notifications*/
                  gaspi_group_t group, /* group through which to gather information*/
                  gaspi_timeout_t timeout /* timeout in ms for blocking operations*/
                  ){
    /* Setup */

    gaspi_number_t group_size;
    gaspi_rank_t *group_ranks;

    GPI_SAFE( gaspi_group_size(group, &group_size) )

    group_ranks = (gaspi_rank_t*) malloc(group_size * sizeof(gaspi_rank_t));

    GPI_SAFE( gaspi_group_ranks(GASPI_GROUP_ALL, group_ranks) )

    /* Exchange information */

    //TODO sanity check to ensure connected with all other members


    gaspi_rank_t irank;
    GPI_SAFE( gaspi_proc_rank(&irank) )

    //find irank index and min rank values in the list
    //may be best to do this in an initialisation step, and store generic useful group values globally (and by global I mean local to the process global) 
    int i;
    int irank_idx=0;
    int min_rank =__UINT32_MAX__;

    for(i=0;i<group_size;i++){
        int cur_rank=group_ranks[i];
        if( cur_rank==irank) irank_idx=i;
        if( cur_rank<min_rank) min_rank=cur_rank;
    }
    
    //remote offset calculation
    offset_remote = offset_remote + irank_idx * size;


    //TODO rewrite for gaspi_write_list for performance improvements
    for(i=0;i<group_size;i++){
        gaspi_rank_t remote_rank = group_ranks[i];
        //if(remote_rank==irank) continue; //TODO update logic to write locally instead? see if it's faster...

#ifdef VERBOSE
        printf("Proc %d sending to %d.\n", irank, remote_rank);
#endif
        GPI_QUEUE_SAFE(
          gaspi_write_notify(segment_id_local,
                           offset_local,
                           remote_rank,
                           segment_id_remote,
                           offset_remote,
                           size,
                           irank,
                           1,
                           queue,
                           timeout)
                           , queue )
    }
    

    //Wait for notifications
    
    gaspi_notification_id_t id;

    for(i=0;i<group_size;i++){ //Need this, as gaspi_notify_waitsome will return on the first non-zero ID.
        GPI_SAFE( gaspi_notify_waitsome(segment_id_remote,
                            min_rank,
                            group_size,
                            &id,
                            timeout) )

        gaspi_notification_t val =0;

        GPI_SAFE( gaspi_notify_reset (segment_id_remote
				, id
				, &val) );

    }

    
    free(group_ranks);

    return GASPI_SUCCESS;
}
