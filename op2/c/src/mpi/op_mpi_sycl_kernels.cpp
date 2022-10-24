/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the OP2 distribution.
 *
 * Copyright (c) 2011, Mike Giles and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Mike Giles may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Mike Giles ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Mike Giles BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define MPICH_IGNORE_CXX_SEEK
#include <op_sycl_rt_support.h>
#include <op_lib_mpi.h>

#include <op_lib_c.h>

void export_halo_gather(cl::sycl::nd_item<1> item, 
                                   int *list, char *dat, int copy_size,
                                   int elem_size, char *export_buffer) {
  int id = item.get_global_id()[0];
  if (id < copy_size) {
    int off = 0;
    if (elem_size % 8 == 0) {
      off += 8 * (elem_size / 8);
      for (int i = 0; i < elem_size / 8; i++) {
        ((double *)(export_buffer + id * elem_size))[i] =
            ((double *)(dat + list[id] * elem_size))[i];
      }
    }
    for (int i = off; i < elem_size; i++) {
      export_buffer[id * elem_size + i] = dat[list[id] * elem_size + i];
    }
  }
}

void export_halo_gather_soa(cl::sycl::nd_item<1> item, 
                                       int *list, char *dat, int copy_size,
                                       int elem_size, char *export_buffer,
                                       int set_size, int dim) {
  int id = item.get_global_id()[0];
  int size_of = elem_size / dim;
  if (id < copy_size) {
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(export_buffer + id * elem_size))[i] =
            ((double *)(dat + list[id] * size_of))[i * set_size];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          export_buffer[id * elem_size + i * size_of + j] =
              dat[list[id] * size_of + i * set_size * size_of + j];
        }
      }
    }
  }
}

void import_halo_scatter_soa(cl::sycl::nd_item<1> item, 
                                        int offset, char *dat, int copy_size,
                                        int elem_size, char *import_buffer,
                                        int set_size, int dim) {
  int id = item.get_global_id()[0];
  int size_of = elem_size / dim;
  if (id < copy_size) {
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + (offset + id) * size_of))[i * set_size] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[(offset + id) * size_of + i * set_size * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

void import_halo_scatter_partial_soa(cl::sycl::nd_item<1> item,
                                                int *list, char *dat,
                                                int copy_size, int elem_size,
                                                char *import_buffer,
                                                int set_size, int dim) {
  int id = item.get_global_id()[0];
  int size_of = elem_size / dim;
  if (id < copy_size) {
    int element = list[id];
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + (element)*size_of))[i * set_size] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[(element)*size_of + i * set_size * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

void import_halo_scatter_partial(cl::sycl::nd_item<1> item,
                                            int *list, char *dat, int copy_size,
                                            int elem_size, char *import_buffer,
                                            int dim) {
  int id = item.get_global_id()[0];
  int size_of = elem_size / dim;
  if (id < copy_size) {
    int element = list[id];
    if (size_of == 8) {
      for (int i = 0; i < dim; i++) {
        ((double *)(dat + element * elem_size))[i] =
            ((double *)(import_buffer + id * elem_size))[i];
      }
    } else {
      for (int i = 0; i < dim; i++) {
        for (int j = 0; j < size_of; j++) {
          dat[element * elem_size + i * size_of + j] =
              import_buffer[id * elem_size + i * size_of + j];
        }
      }
    }
  }
}

void gather_data_to_buffer(op_arg arg, halo_list exp_exec_list,
                           halo_list exp_nonexec_list) {
  int threads = 192;
  int blocks = 1 + ((exp_exec_list->size - 1) / 192);

  int *eel_d = export_exec_list_d[arg.dat->set->index];
  int *enl_d = export_nonexec_list_d[arg.dat->set->index];
  char *data_d = arg.data_d;
  int eel_size = exp_exec_list->size;
  int enl_size = exp_nonexec_list->size;
  int dat_size = arg.dat->size;
  char *buffer_d = arg.dat->buffer_d;
  int datdim = arg.dat->dim;
  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;

    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
           export_halo_gather_soa(item,
              eel_d, data_d,
              eel_size, dat_size, buffer_d, set_size,
              datdim);   
              });  
        });

    int blocks2 = 1 + ((exp_nonexec_list->size - 1) / 192);
    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather_soa2>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks2 * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
           export_halo_gather_soa(item,
              enl_d, data_d,
              enl_size, dat_size,
              buffer_d + eel_size * dat_size, set_size,
              datdim);  
              });  
        });
  } else {
    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
           export_halo_gather(item,
              eel_d, data_d,
              eel_size, dat_size, buffer_d);   
              });  
        });

    int blocks2 = 1 + ((exp_nonexec_list->size - 1) / 192);
    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather2>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks2 * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
           export_halo_gather(item,
              enl_d, data_d,
              enl_size, dat_size,
              buffer_d + eel_size * dat_size);   
              });  
        });
  }
}

void gather_data_to_buffer_partial(op_arg arg, halo_list exp_nonexec_list) {
  int threads = 192;
  int blocks = 1 + ((exp_nonexec_list->size - 1) / 192);
  int *enlp_d = export_nonexec_list_partial_d[arg.map->index];
  char *data_d = arg.data_d;

  int enl_size = exp_nonexec_list->size;
  int dat_size = arg.dat->size;
  char *buffer_d = arg.dat->buffer_d;
  int datdim = arg.dat->dim;

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;

    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather_soa3>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               export_halo_gather_soa(item,
                  enlp_d, data_d,
                  enl_size, dat_size, buffer_d, set_size,
                  datdim);
              });  
        });

  } else {
        op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_export_halo_gather3>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               export_halo_gather(item,
                  enlp_d, data_d,
                  enl_size, dat_size, buffer_d);
              });  
        });
  }
}

void scatter_data_from_buffer(op_arg arg) {
  int threads = 192;
  int blocks = 1 + ((arg.dat->set->exec_size - 1) / 192);
  char *buffer_d_r = arg.dat->buffer_d_r;
  int datdim = arg.dat->dim;
  int datsize = arg.dat->size;
  char *data_d = arg.dat->data_d;
  int exec_size = arg.dat->set->exec_size;

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;
    int offset = arg.dat->set->size;
    int copy_size = arg.dat->set->exec_size;

    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_import_halo_scatter_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               import_halo_scatter_soa(item,
                  offset, data_d, copy_size, datsize, buffer_d_r,
                  set_size, datdim);
              });  
        });
    

    offset += arg.dat->set->exec_size;
    copy_size = arg.dat->set->nonexec_size;

    int blocks2 = 1 + ((arg.dat->set->nonexec_size - 1) / 192);
    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_import_halo_scatter_soa2>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks2 * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               import_halo_scatter_soa(item,
                  offset, data_d, copy_size, datsize,
                  buffer_d_r + exec_size * datsize, set_size,
                  datdim);
              });  
        });
    
  }
}

void scatter_data_from_buffer_partial(op_arg arg) {
  int threads = 192;
  int blocks = 1 + ((OP_import_nonexec_permap[arg.map->index]->size - 1) / 192);
  int datsize = arg.dat->size;
  char *buffer_d = arg.dat->buffer_d;
  int datdim = arg.dat->dim;
  char *data_d = arg.dat->data_d;
  int exec_size = arg.dat->set->exec_size;
  int *inlp_d = import_nonexec_list_partial_d[arg.map->index];

  if (strstr(arg.dat->type, ":soa") != NULL ||
      (OP_auto_soa && arg.dat->dim > 1)) {

    int set_size = arg.dat->set->size + arg.dat->set->exec_size +
                   arg.dat->set->nonexec_size;
    int init = OP_export_nonexec_permap[arg.map->index]->size;
    int copy_size = OP_import_nonexec_permap[arg.map->index]->size;

    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_import_halo_scatter_partial_soa>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               import_halo_scatter_partial_soa(item,
                  inlp_d, data_d, copy_size,
                  datsize, buffer_d + init * datsize, set_size,
                  datdim);
              });  
        });
    
  } else {
    int init = OP_export_nonexec_permap[arg.map->index]->size;
    int copy_size = OP_import_nonexec_permap[arg.map->index]->size;

    op2_queue->submit(
        [&](cl::sycl::handler &cgh) {
          cgh.parallel_for<class ops_import_halo_scatter_partial_soa2>(
              cl::sycl::nd_range<1>(
                  cl::sycl::range<1>(blocks * threads),
                  cl::sycl::range<1>(threads)),
              [=](cl::sycl::nd_item<1> item) {
               import_halo_scatter_partial(item,
                  inlp_d, data_d, copy_size,
                  datsize, buffer_d + init * datsize, datdim);
              });  
        });
    
  }
}
