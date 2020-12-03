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

//
// This file implements the CUDA-specific run-time support functions
//

//
// header files
//

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include <CL/sycl.hpp>
//#include <math_constants.h>

#include <op_sycl_rt_support.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

// arrays for global constants and reductions

int OP_consts_bytes = 0, OP_reduct_bytes = 0;

char *OP_consts_h=NULL, *OP_consts_d=NULL, *OP_reduct_h=NULL, *OP_reduct_d=NULL;

cl::sycl::queue *op2_queue=NULL;

void op_mvHostToDevice(void **map, int size, const char* type) {
  if (!OP_hybrid_gpu)
    return;

  if (strstr(type, "double") != NULL) {
    size = size / sizeof(double);
    auto *temp = new cl::sycl::buffer<double, 1>(cl::sycl::range<1>(size));
    void *tmp = (void*) temp;
    double *data = (double*)(*map);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++)
        HostAccessor[i] = data[i];
    #endif
    free(*map);
    *map = tmp;

  } else if (strstr(type, "float") != NULL) {
    size = size / sizeof(float);
    auto *temp = new cl::sycl::buffer<float, 1>(cl::sycl::range<1>(size));
    void *tmp = (void*) temp;
    float *data = (float*)(*map);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++)
        HostAccessor[i] = data[i];
    #endif
    free(*map);
    *map = tmp;

  } else if (strstr(type, "int") != NULL) {
    size = size / sizeof(int);
    auto *temp = new cl::sycl::buffer<int, 1>(cl::sycl::range<1>(size));
    void *tmp = (void*) temp;
    int *data = (int*)(*map);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++)
        HostAccessor[i] = data[i];
    #endif
    free(*map);
    *map = tmp;

  } else if (strstr(type, "short") != NULL) {
    size = size / sizeof(short);
    auto *temp = new cl::sycl::buffer<short, 1>(cl::sycl::range<1>(size));
    void *tmp = (void*) temp;
    short *data = (short*)(*map);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++)
        HostAccessor[i] = data[i];
    #endif
    free(*map);
    *map = tmp;

  } else {
    op_printf("Error: op_mvHostToDevice() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

void op_cpHostToDevice(void **data_d, void **data_h, int size, const char* type) {
  if (!OP_hybrid_gpu)
    return;

  if (strstr(type, "double") != NULL) {
    size = size / sizeof(double);
    if (*data_d != NULL)
      delete static_cast<cl::sycl::buffer<double, 1> *>((void*)*data_d);
    auto *buf =  new cl::sycl::buffer<double, 1>(cl::sycl::range<1>(size));
    *data_d = (void*)buf;
    double *data = (double*)(*data_h);

    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*buf).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*buf).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++) HostAccessor[i] = data[i];
    #endif

  } else if (strstr(type, "float") != NULL) {
    size = size / sizeof(float);
    if (*data_d != NULL)
      delete static_cast<cl::sycl::buffer<float, 1> *>((void*)*data_d);
    auto *buf =  new cl::sycl::buffer<float, 1>(cl::sycl::range<1>(size));
    *data_d = (void*)buf;
    float *data = (float*)(*data_h);

    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*buf).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*buf).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++) HostAccessor[i] = data[i];
    #endif

  } else if (strstr(type, "int") != NULL) {
    size = size / sizeof(int);
    if (*data_d != NULL)
      delete static_cast<cl::sycl::buffer<int, 1> *>((void*)*data_d);
    auto *buf =  new cl::sycl::buffer<int, 1>(cl::sycl::range<1>(size));
    *data_d = (void*)buf;
    int *data = (int*)(*data_h);

    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*buf).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy(data, acc);
      });
      op2_queue->wait();
    #else
      auto HostAccessor = (*buf).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < size; i++) HostAccessor[i] = data[i];
    #endif

  } else {
    op_printf("Error: op_cpHostToDevice() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

op_plan *op_plan_get(char const *name, op_set set, int part_size, int nargs,
                     op_arg *args, int ninds, int *inds) {
  return op_plan_get_stage(name, set, part_size, nargs, args, ninds, inds,
                           OP_STAGE_ALL);
}

op_plan *op_plan_get_stage(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging) {
  return op_plan_get_stage_upload(name, set, part_size, nargs, args, ninds, inds,
                           staging, 1);
}

op_plan *op_plan_get_stage_upload(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging, int upload) {
  op_plan *plan =
      op_plan_core(name, set, part_size, nargs, args, ninds, inds, staging);
  if (!OP_hybrid_gpu || !upload)
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
    if (offsets[plan->ninds_staged]>0)
      op_mvHostToDevice((void **)&(plan->ind_map),
                                offsets[plan->ninds_staged] * set_size * sizeof(int), "int");
    else plan->ind_map = NULL;
    for (int m = 0; m < plan->ninds_staged; m++) {
      plan->ind_maps[m] = NULL; // &plan->ind_map[set_size * offsets[m]];
    }

    int counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL)
        counter++;
    short *tmp2 = plan->loc_map;
    if (counter > 0)
      op_mvHostToDevice((void **)&(plan->loc_map),
                                sizeof(short) * counter * set_size, "short");
    else plan->loc_map = NULL;
    counter = 0;
    for (int m = 0; m < nargs; m++)
      if (plan->loc_maps[m] != NULL) {
        plan->loc_maps[m] = NULL; //&plan->loc_map[set_size * counter];
        counter++;
      }

    if (plan->ninds_staged>0)
      op_mvHostToDevice((void **)&(plan->ind_sizes),
          sizeof(int) * plan->nblocks * plan->ninds_staged, "int");
    else plan->ind_sizes = NULL;
    if (plan->ninds_staged>0)
      op_mvHostToDevice((void **)&(plan->ind_offs),
                                sizeof(int) * plan->nblocks * plan->ninds_staged, "int");
    else plan->ind_offs = NULL;
    op_mvHostToDevice((void **)&(plan->nthrcol), sizeof(int) * plan->nblocks, "int");
    op_mvHostToDevice((void **)&(plan->thrcol), sizeof(int) * set_size, "int");
    op_mvHostToDevice((void **)&(plan->col_reord), sizeof(int) * set_size, "int");
    op_mvHostToDevice((void **)&(plan->offset), sizeof(int) * plan->nblocks, "int");
    plan->offset_d = plan->offset;
    op_mvHostToDevice((void **)&(plan->nelems), sizeof(int) * plan->nblocks, "int");
    plan->nelems_d = plan->nelems;
    op_mvHostToDevice((void **)&(plan->blkmap), sizeof(int) * plan->nblocks, "int");
    plan->blkmap_d = plan->blkmap;
  }

  return plan;
}

std::vector<void*> op_sycl_consts;

void op_sycl_exit() {
  if (!OP_hybrid_gpu)
    return;
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)(item->dat)->data_d);
  }

  for (int i = 0; i < OP_map_index; i++)
    delete static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_map_list[i]->map_d);

  for (int ip = 0; ip < OP_plan_index; ip++) {
    if (OP_plans[ip].ind_map!=NULL) delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].ind_map); OP_plans[ip].ind_map = NULL;
    if (OP_plans[ip].loc_map!= NULL) delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].loc_map); OP_plans[ip].loc_map = NULL;
    if (OP_plans[ip].ind_sizes!=NULL) delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].ind_sizes); OP_plans[ip].ind_sizes = NULL;
    if (OP_plans[ip].ind_offs!=NULL) delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].ind_offs); OP_plans[ip].ind_offs = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].nthrcol); OP_plans[ip].nthrcol = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].thrcol); OP_plans[ip].thrcol = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].col_reord); OP_plans[ip].col_reord = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].offset); OP_plans[ip].offset = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].nelems); OP_plans[ip].nelems = NULL;
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_plans[ip].blkmap); OP_plans[ip].blkmap = NULL;
  }
  if (OP_consts_bytes > 0) {
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_consts_d);
  }
  if (OP_reduct_bytes > 0) {
    delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_reduct_d);
  }
  for (size_t i = 0; i < op_sycl_consts.size(); i++)
    delete static_cast<cl::sycl::buffer<char,1>*>(op_sycl_consts[i]);

  delete op2_queue;
}


void *op_sycl_register_const(void *old_p, void *new_p) {
  if (old_p == NULL) op_sycl_consts.push_back(new_p);
  else {
    for (size_t i = 0; i < op_sycl_consts.size(); i++)
      if (op_sycl_consts[i]==old_p) {
        delete static_cast<cl::sycl::buffer<char,1>*>(op_sycl_consts[i]);
        op_sycl_consts[i]=new_p;
      }
  }
  return new_p;
}
//
// routines to resize constant/reduct arrays, if necessary
//
/*
// UPDATE: sycl::buffer must be type-specific, no more 'generic' 
//         sycl::buffer<char>. This rules out realloc methods on a
//         single global buffer object.
void reallocConstArrays(int consts_bytes) {
  if (consts_bytes > OP_consts_bytes) {
    if (OP_consts_bytes > 0) {
      delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_consts_d);
      free(OP_consts_h);
    }
    OP_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
    OP_consts_d = (char*)(void*)new cl::sycl::buffer<char, 1>(
            cl::sycl::range<1>(OP_consts_bytes));
    OP_consts_h = (char*)op_malloc(OP_consts_bytes);
  }
}

void reallocReductArrays(int reduct_bytes) {
  if (reduct_bytes > OP_reduct_bytes) {
    if (OP_reduct_bytes > 0) {
      delete static_cast<cl::sycl::buffer<char, 1> *>((void*)OP_reduct_d);
      free(OP_reduct_h);
    }
    OP_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
    OP_reduct_d = (char*)(void*)new cl::sycl::buffer<char, 1>(
            cl::sycl::range<1>(OP_reduct_bytes));
    OP_reduct_h = (char*)op_malloc(OP_reduct_bytes);
  }
}
*/

//
// routines to alloc/free constant/reduct arrays
//
void allocConstArrays(int consts_bytes, const char* type) {
  if (OP_consts_d != NULL) {
    op_printf("Error: allocConstArrays() - OP_consts_d must be NULL\n");
    exit(-1);
  }

  OP_consts_bytes = 4 * consts_bytes; // 4 is arbitrary, more than needed
  if (strstr(type, "double") != NULL) {
    OP_consts_d = (char*)(void*)new cl::sycl::buffer<double, 1>(cl::sycl::range<1>(OP_consts_bytes/sizeof(double)));
  } else if (strstr(type, "float") != NULL) {
    OP_consts_d = (char*)(void*)new cl::sycl::buffer<float, 1>(cl::sycl::range<1>(OP_consts_bytes/sizeof(float)));
  } else if (strstr(type, "int") != NULL) {
    OP_consts_d = (char*)(void*)new cl::sycl::buffer<int, 1>(cl::sycl::range<1>(OP_consts_bytes/sizeof(int)));
  } else {
    op_printf("Error: allocConstArrays() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
  
  OP_consts_h = (char*)op_malloc(OP_consts_bytes);
}
void freeConstArrays(const char* type) {
  if (OP_consts_d == NULL)
    return;

  if (OP_consts_d != NULL) {
    if (strstr(type, "double") != NULL) {
      delete static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_consts_d);
    } else if (strstr(type, "float") != NULL) {
      delete static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_consts_d);
    } else if (strstr(type, "int") != NULL) {
      delete static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_consts_d);
    } else {
      op_printf("Error: freeConstArrays() - unrecognised data type '%s'\n", type);
      exit(-1);
    }
    OP_consts_d = NULL;

    free(OP_consts_h); OP_consts_h=NULL;
  }
}

void allocReductArrays(int reduct_bytes, const char* type) {
  if (OP_reduct_d != NULL) {
    op_printf("Error: allocReductArrays() - OP_reduct_d must be NULL\n");
    exit(-1);
  }

  OP_reduct_bytes = 4 * reduct_bytes; // 4 is arbitrary, more than needed
  if (strstr(type, "double") != NULL) {
    OP_reduct_d = (char*)(void*)new cl::sycl::buffer<double, 1>(cl::sycl::range<1>(OP_reduct_bytes/sizeof(double)));
  } else if (strstr(type, "float") != NULL) {
    OP_reduct_d = (char*)(void*)new cl::sycl::buffer<float, 1>(cl::sycl::range<1>(OP_reduct_bytes/sizeof(float)));
  } else if (strstr(type, "int") != NULL) {
    OP_reduct_d = (char*)(void*)new cl::sycl::buffer<int, 1>(cl::sycl::range<1>(OP_reduct_bytes/sizeof(int)));
  } else {
    op_printf("Error: reallocReductArrays() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
  OP_reduct_h = (char*)op_malloc(OP_reduct_bytes);
}
void freeReductArrays(const char* type) {
  if (OP_reduct_d == NULL)
    return;

  if (strstr(type, "double") != NULL) {
    delete static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_reduct_d);
  } else if (strstr(type, "float") != NULL) {
    delete static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_reduct_d);
  } else if (strstr(type, "int") != NULL) {
    delete static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_reduct_d);
  } else {
    op_printf("Error: freeReductArrays() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
  OP_reduct_d = NULL;
  
  free(OP_reduct_h); OP_reduct_h=NULL;
}

//
// routines to move constant/reduct arrays
//

void mvConstArraysToDevice(int consts_bytes, const char* type) {
  if (OP_consts_d == NULL) {
    op_printf("Error: mvConstArraysToDevice() - cannot move NULL pointer\n");
    exit(-1);
  }

  if (strstr(type, "double") != NULL) {
    consts_bytes = consts_bytes/sizeof(double);
    cl::sycl::buffer<double, 1> *temp = static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((double*)OP_consts_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < consts_bytes; i++)
        HostAccessor[i] = ((double*)OP_consts_h)[i];
    #endif

  } else if (strstr(type, "float") != NULL) {
    consts_bytes = consts_bytes/sizeof(float);
    cl::sycl::buffer<float, 1> *temp = static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((float*)OP_consts_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < consts_bytes; i++)
        HostAccessor[i] = ((float*)OP_consts_h)[i];
    #endif

  } else if (strstr(type, "int") != NULL) {
    consts_bytes = consts_bytes/sizeof(int);
    cl::sycl::buffer<int, 1> *temp = static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((int*)OP_consts_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < consts_bytes; i++)
        HostAccessor[i] = ((int*)OP_consts_h)[i];
    #endif

  } else {
    op_printf("Error: mvConstArraysToDevice() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

void mvConstArraysToHost(int consts_bytes, const char* type) {
  if (OP_consts_d == NULL) {
    op_printf("Error: mvConstArraysToHost() - cannot move NULL pointer\n");
    exit(-1);
  }

  if (strstr(type, "double") != NULL) {
    consts_bytes = consts_bytes/sizeof(double);
    cl::sycl::buffer<double, 1> *temp = static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(double*)OP_consts_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < consts_bytes; i++)
        ((double*)OP_consts_h)[i] = HostAccessor[i];
    #endif

  } else if (strstr(type, "float") != NULL) {
    consts_bytes = consts_bytes/sizeof(float);
    cl::sycl::buffer<float, 1> *temp = static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(float*)OP_consts_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < consts_bytes; i++)
        ((float*)OP_consts_h)[i] = HostAccessor[i];
    #endif

  } else if (strstr(type, "int") != NULL) {
    consts_bytes = consts_bytes/sizeof(int);
    cl::sycl::buffer<int, 1> *temp = static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_consts_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(int*)OP_consts_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < consts_bytes; i++)
        ((int*)OP_consts_h)[i] = HostAccessor[i];
    #endif

  } else {
    op_printf("Error: mvConstArraysToHost() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

void mvReductArraysToDevice(int reduct_bytes, const char* type) {
  if (OP_reduct_d == NULL) {
    op_printf("Error: mvReductArraysToDevice() - cannot move NULL pointer\n");
    exit(-1);
  }

  if (strstr(type, "double") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(double);
    cl::sycl::buffer<double, 1> *temp = static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((double*)OP_reduct_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < reduct_bytes; i++)
        HostAccessor[i] = ((double*)OP_reduct_h)[i];
    #endif

  } else if (strstr(type, "float") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(float);
    cl::sycl::buffer<float, 1> *temp = static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((float*)OP_reduct_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < reduct_bytes; i++)
        HostAccessor[i] = ((float*)OP_reduct_h)[i];
    #endif

  } else if (strstr(type, "int") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(int);
    cl::sycl::buffer<int, 1> *temp = static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
          cgh.copy((int*)OP_reduct_h, acc);
          });
    #else
      auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
      for (size_t i = 0; i < reduct_bytes; i++)
        HostAccessor[i] = ((int*)OP_reduct_h)[i];
    #endif

  } else {
    op_printf("Error: mvReductArraysToDevice() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

void mvReductArraysToHost(int reduct_bytes, const char* type) {
  if (OP_reduct_d == NULL) {
    op_printf("Error: mvReductArraysToHost() - cannot move NULL pointer\n");
    exit(-1);
  }

  if (strstr(type, "double") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(double);
    cl::sycl::buffer<double, 1> *temp = static_cast<cl::sycl::buffer<double, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(double*)OP_reduct_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < reduct_bytes; i++)
        ((double*)OP_reduct_h)[i] = HostAccessor[i];
    #endif

  } else if (strstr(type, "float") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(float);
    cl::sycl::buffer<float, 1> *temp = static_cast<cl::sycl::buffer<float, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(float*)OP_reduct_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < reduct_bytes; i++)
        ((float*)OP_reduct_h)[i] = HostAccessor[i];
    #endif

  } else if (strstr(type, "int") != NULL) {
    reduct_bytes = reduct_bytes/sizeof(int);
    cl::sycl::buffer<int, 1> *temp = static_cast<cl::sycl::buffer<int, 1> *>((void*)OP_reduct_d);
    #ifdef SYCL_COPY
      op2_queue->submit([&](cl::sycl::handler &cgh) {
          auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
          cgh.copy(acc,(int*)OP_reduct_h);
          });
        op2_queue->wait();
    #else
      const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
      for (size_t i = 0; i < reduct_bytes; i++)
        ((int*)OP_reduct_h)[i] = HostAccessor[i];
    #endif

  } else {
    op_printf("Error: mvReductArraysToHost() - unrecognised data type '%s'\n", type);
    exit(-1);
  }
}

//
// routine to fetch data from GPU to CPU (with transposing SoA to AoS if needed)
//

void op_sycl_get_data(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  if (dat->dirty_hd == 2)
    dat->dirty_hd = 0;
  else
    return;
  // transpose data
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    cl::sycl::accessor<char, 1, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>
      temp_data(*static_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < dat->set->size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              temp_data[element_size * i * dat->set->size + element_size * j +
                        c];
        }
      }
    }
  } else {
    cl::sycl::buffer<char, 1> *temp = static_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d);
#ifdef SYCL_COPY
    op2_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc,dat->data);
      });
    op2_queue->wait();
#else
  const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < dat->set->size*dat->size; i++)
    dat->data[i] = HostAccessor[i];
#endif
  }
}

void deviceSync() {
  op2_queue->wait(); 
}

#ifndef OPMPI
extern int OP_sycl_device;
void syclDeviceInit(int argc, char **argv) {
  (void)argc;
  (void)argv;
  cl::sycl::device_selector *selector;
  switch (OP_sycl_device) {
    case 0:
      selector = new cl::sycl::host_selector();
      break;
    case 1:
      selector = new cl::sycl::cpu_selector();
      break;
    case 2:
      selector = new cl::sycl::gpu_selector();
      break;
    case 3:
      selector = new cl::sycl::default_selector();
      break;
    default:
      op_printf("Error, unrecognised SYCL device selection\n");
      exit(-1);
  }
  op2_queue = new cl::sycl::queue(*selector);
  delete selector;
  OP_hybrid_gpu = 1;
  std::cout << "Running on " << op2_queue->get_device().get_info<cl::sycl::info::device::name>() << ", cores: "<< op2_queue->get_device().get_info<cl::sycl::info::device::max_compute_units>() <<"\n";
}

void op_upload_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  int set_size = dat->set->size;
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    cl::sycl::accessor<char, 1, cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>
      temp_data(*static_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          temp_data[element_size * i * set_size + element_size * j + c] =
              dat->data[dat->size * j + element_size * i + c];
        }
      }
    }
  } else {
    cl::sycl::buffer<char, 1> *temp = static_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d);
#ifdef SYCL_COPY
    op2_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc = (*temp).template get_access<cl::sycl::access::mode::write>(cgh);
      cgh.copy(dat->data,acc);
      });
#else
  auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::write>();
  for (size_t i = 0; i < set_size * dat->size; i++)
    HostAccessor[i] = dat->data[i];
#endif
  }
}

void op_download_dat(op_dat dat) {
  if (!OP_hybrid_gpu)
    return;
  int set_size = dat->set->size;
  // transpose data
  if (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) {
    cl::sycl::accessor<double, 1, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>
      temp_data(*static_cast<cl::sycl::buffer<double, 1> *>((void*)dat->data_d));
    int element_size = dat->size / dat->dim;
    for (int i = 0; i < dat->dim; i++) {
      for (int j = 0; j < set_size; j++) {
        for (int c = 0; c < element_size; c++) {
          dat->data[dat->size * j + element_size * i + c] =
              temp_data[element_size * i * set_size + element_size * j +
                        c];
        }
      }
    }
  } else {
    cl::sycl::buffer<char, 1> *temp = static_cast<cl::sycl::buffer<char, 1> *>((void*)dat->data_d);
#ifdef SYCL_COPY
    op2_queue->submit([&](cl::sycl::handler &cgh) {
        auto acc = (*temp).template get_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc,dat->data);
      });
    op2_queue->wait();
#else
  const auto HostAccessor = (*temp).get_access<cl::sycl::access::mode::read>();
  for (size_t i = 0; i < dat->set->size*dat->size; i++)
    dat->data[i] = HostAccessor[i];
#endif
  }
}

int op_mpi_halo_exchanges(op_set set, int nargs, op_arg *args) {
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

void op_mpi_wait_all(int nargs, op_arg *args) {
  (void)nargs;
  (void)args;
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
void op_compute_moment_across_times(double* times, int ntimes, bool ignore_zeros, double *first, double *second) {
  *first = 0.0;
  *second = 0.0f;
  int n = 0;
  for (int i=0; i<ntimes; i++) {
    if (ignore_zeros && (times[i] == 0.0f)) {
      continue;
    }
    *first += times[i];
    *second += times[i] * times[i];
    n++;
  }

  if (n != 0) {
    *first /= (double)n;
    *second /= (double)n;
  }
}

int op_is_root() { return 1; }
#endif
