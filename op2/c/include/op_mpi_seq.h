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

#ifndef __OP_MPI_SEQ_H
#define __OP_MPI_SEQ_H

/*
 * op_mpi_seq.h
 *
 * Headder file declares and defines the OP2 Distributed memory (MPI)
 * op_par_loop runctions
 *
 * intra-node - sequential execution
 * inter-node - MPI execution
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

#include <op_lib_core.h>
#include <op_rt_support.h>

/*******************************************************************************
* Random partitioning wrapper prototype
*******************************************************************************/
void op_partition_random(op_set primary_set);

inline void op_arg_set(int n, op_arg arg, char **p_arg)
{
  int n2;
  if (arg.map==NULL)         // identity mapping, or global data
    n2 = n;
  else                       // standard pointers
    n2 = arg.map->map[arg.idx+n*arg.map->dim];

  *p_arg = arg.data + n2*arg.size;
}

inline op_arg* blank_arg(op_arg *arg)
{
  op_arg *junck = NULL;
  if(arg->argtype == OP_ARG_GBL && //this argument is OP_GBL and
      arg->acc != OP_READ)    //OP_INC or OP_MAX/MIN
  {
    return junck;
  }
  else
  {
    return arg;
  }
}

/*******************************************************************************
* op_par_loop template for 2 arguments
*******************************************************************************/

template < class T0, class T1 >
void op_par_loop(void (*kernel)( T0*, T1* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1 )
{
  char *p_arg0, *p_arg1;
  int exec_length = 0;

  int sent[2] = {0,0};

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if(arg0.idx != -1 || arg1.idx != -1)//indirect loop
  {
    if (OP_diags==1) {
      if(arg0.argtype == OP_ARG_DAT) reset_halo(arg0);
      if(arg1.argtype == OP_ARG_DAT) reset_halo(arg1);
    }

    //for each indirect data set
    if(arg0.argtype == OP_ARG_DAT) sent[0] = exchange_halo(arg0);
    if(arg1.argtype == OP_ARG_DAT) sent[1] = exchange_halo(arg1);
  }

  //if all indirect dataset access is with OP_READ
  exec_length = set->size;
  //else
  if (arg0.idx != -1 && arg0.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg1.idx != -1 && arg1.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;

  // loop over set elements
  //(1) over core partition
  for (int n=0; n<core_num[set->index]; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1);
  }

  //wait for comms to complete
  if(arg0.argtype == OP_ARG_DAT) if(sent[0] == 1 )wait_all(arg0);
  if(arg1.argtype == OP_ARG_DAT) if(sent[1] == 1 )wait_all(arg1);

  for (int n=core_num[set->index]; n<set->size; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1);
  }

  //(2) over exec halo (blank out global parameters to avoid double counting)
  for (int n=set->size; n<exec_length; n++) {
    op_arg_set(n,*(blank_arg(&arg0)),&p_arg0 );
    op_arg_set(n,*(blank_arg(&arg1)),&p_arg1 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1);
  }

  //set dirty bit on direct/indirect datasets with access OP_INC,OP_WRITE, OP_RW
  if(arg0.argtype == OP_ARG_DAT)set_dirtybit(arg0);
  if(arg1.argtype == OP_ARG_DAT)set_dirtybit(arg1);

  //performe any global operations
  if(arg0.argtype == OP_ARG_GBL)
    global_reduce(&arg0);
  if(arg1.argtype == OP_ARG_GBL)
    global_reduce(&arg1);

  //update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  if(sent[0] == 1)op_mpi_perf_comm(k_i, arg0);
  if(sent[1] == 1)op_mpi_perf_comm(k_i, arg1);
#endif
}

/*******************************************************************************
* op_par_loop template for 4 arguments
*******************************************************************************/

template < class T0, class T1, class T2, class T3 >
void op_par_loop(void (*kernel)( T0*, T1*, T2*, T3*),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3)
{
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3;
  int exec_length = 0;
  int sent[4] = {0,0,0,0};

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if(arg0.idx != -1 || arg1.idx != -1 || arg2.idx != -1 || arg3.idx != -1)//indirect loop
  {
    if (OP_diags==1) {
      if(arg0.argtype == OP_ARG_DAT) reset_halo(arg0);
      if(arg1.argtype == OP_ARG_DAT) reset_halo(arg1);
      if(arg2.argtype == OP_ARG_DAT) reset_halo(arg2);
      if(arg3.argtype == OP_ARG_DAT) reset_halo(arg3);
    }

    //for each indirect data set
    if(arg0.argtype == OP_ARG_DAT) sent[0] = exchange_halo(arg0);
    if(arg1.argtype == OP_ARG_DAT) sent[1] = exchange_halo(arg1);
    if(arg2.argtype == OP_ARG_DAT) sent[2] = exchange_halo(arg2);
    if(arg3.argtype == OP_ARG_DAT) sent[3] = exchange_halo(arg3);
  }

  //if all indirect dataset access is with OP_READ
  exec_length = set->size;
  //else
  if (arg0.idx != -1 && arg0.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg1.idx != -1 && arg1.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg2.idx != -1 && arg2.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg3.idx != -1 && arg3.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;

  // loop over set elements
  //(1) over core partition
  for (int n=0; n<core_num[set->index]; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3);
  }

  //wait for comms to complete
  if(arg0.argtype == OP_ARG_DAT) if(sent[0] == 1 )wait_all(arg0);
  if(arg1.argtype == OP_ARG_DAT) if(sent[1] == 1 )wait_all(arg1);
  if(arg2.argtype == OP_ARG_DAT) if(sent[2] == 1 )wait_all(arg2);
  if(arg3.argtype == OP_ARG_DAT) if(sent[3] == 1 )wait_all(arg3);

  for (int n=core_num[set->index]; n<set->size; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3);
  }

  //(2) over exec halo (blank out global parameters to avoid double counting)
  for (int n=set->size; n<exec_length; n++) {
    op_arg_set(n,*(blank_arg(&arg0)),&p_arg0 );
    op_arg_set(n,*(blank_arg(&arg1)),&p_arg1 );
    op_arg_set(n,*(blank_arg(&arg2)),&p_arg2 );
    op_arg_set(n,*(blank_arg(&arg3)),&p_arg3 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3);
  }

  //set dirty bit on direct/indirect datasets with access OP_INC,OP_WRITE, OP_RW
  if(arg0.argtype == OP_ARG_DAT)set_dirtybit(arg0);
  if(arg1.argtype == OP_ARG_DAT)set_dirtybit(arg1);
  if(arg2.argtype == OP_ARG_DAT)set_dirtybit(arg2);
  if(arg3.argtype == OP_ARG_DAT)set_dirtybit(arg3);

  //performe any global operations
  if(arg0.argtype == OP_ARG_GBL)
    global_reduce(&arg0);
  if(arg1.argtype == OP_ARG_GBL)
    global_reduce(&arg1);;
  if(arg2.argtype == OP_ARG_GBL)
    global_reduce(&arg2);
  if(arg3.argtype == OP_ARG_GBL)
    global_reduce(&arg3);

  //update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  if(sent[0] == 1)op_mpi_perf_comm(k_i, arg0);
  if(sent[1] == 1)op_mpi_perf_comm(k_i, arg1);
  if(sent[2] == 1)op_mpi_perf_comm(k_i, arg2);
  if(sent[3] == 1)op_mpi_perf_comm(k_i, arg3);
#endif
}

/*******************************************************************************
* op_par_loop template for 5 arguments
*******************************************************************************/

template < class T0, class T1, class T2, class T3,
           class T4 >
void op_par_loop(void (*kernel)( T0*, T1*, T2*, T3*,
                                 T4* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4 )
{
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4;
  int exec_length = 0;
  int sent[5] = {0,0,0,0,0};

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if(arg0.idx != -1 || arg1.idx != -1 || arg2.idx != -1 || arg3.idx != -1 ||
      arg4.idx != -1 )//indirect loop
  {
    if (OP_diags==1) {
      if(arg0.argtype == OP_ARG_DAT) reset_halo(arg0);
      if(arg1.argtype == OP_ARG_DAT) reset_halo(arg1);
      if(arg2.argtype == OP_ARG_DAT) reset_halo(arg2);
      if(arg3.argtype == OP_ARG_DAT) reset_halo(arg3);
      if(arg4.argtype == OP_ARG_DAT) reset_halo(arg4);
    }

    //for each indirect data set
    if(arg0.argtype == OP_ARG_DAT) sent[0] = exchange_halo(arg0);
    if(arg1.argtype == OP_ARG_DAT) sent[1] = exchange_halo(arg1);
    if(arg2.argtype == OP_ARG_DAT) sent[2] = exchange_halo(arg2);
    if(arg3.argtype == OP_ARG_DAT) sent[3] = exchange_halo(arg3);
    if(arg4.argtype == OP_ARG_DAT) sent[4] = exchange_halo(arg4);
  }

  //if all indirect dataset access is with OP_READ
  exec_length = set->size;
  //else
  if (arg0.idx != -1 && arg0.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg1.idx != -1 && arg1.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg2.idx != -1 && arg2.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg3.idx != -1 && arg3.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg4.idx != -1 && arg4.acc != OP_READ)
      exec_length = set->size + OP_import_exec_list[set->index]->size;

  // loop over set elements
  //(1) over core partition
  for (int n=0; n<core_num[set->index]; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4);
  }

  //wait for comms to complete
  if(arg0.argtype == OP_ARG_DAT) if(sent[0] == 1 )wait_all(arg0);
  if(arg1.argtype == OP_ARG_DAT) if(sent[1] == 1 )wait_all(arg1);
  if(arg2.argtype == OP_ARG_DAT) if(sent[2] == 1 )wait_all(arg2);
  if(arg3.argtype == OP_ARG_DAT) if(sent[3] == 1 )wait_all(arg3);
  if(arg4.argtype == OP_ARG_DAT) if(sent[4] == 1 )wait_all(arg4);

  for (int n=core_num[set->index]; n<set->size; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4);
  }

  //(2) over exec halo (blank out global parameters to avoid double counting)
  for (int n=set->size; n<exec_length; n++) {
    op_arg_set(n,*(blank_arg(&arg0)),&p_arg0 );
    op_arg_set(n,*(blank_arg(&arg1)),&p_arg1 );
    op_arg_set(n,*(blank_arg(&arg2)),&p_arg2 );
    op_arg_set(n,*(blank_arg(&arg3)),&p_arg3 );
    op_arg_set(n,*(blank_arg(&arg4)),&p_arg4 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4);
  }

  //set dirty bit on direct/indirect datasets with access OP_INC,OP_WRITE, OP_RW
  if(arg0.argtype == OP_ARG_DAT)set_dirtybit(arg0);
  if(arg1.argtype == OP_ARG_DAT)set_dirtybit(arg1);
  if(arg2.argtype == OP_ARG_DAT)set_dirtybit(arg2);
  if(arg3.argtype == OP_ARG_DAT)set_dirtybit(arg3);
  if(arg4.argtype == OP_ARG_DAT)set_dirtybit(arg4);

  //performe any global operations
  if(arg0.argtype == OP_ARG_GBL)
    global_reduce(&arg0);
  if(arg1.argtype == OP_ARG_GBL)
    global_reduce(&arg1);;
  if(arg2.argtype == OP_ARG_GBL)
    global_reduce(&arg2);
  if(arg3.argtype == OP_ARG_GBL)
    global_reduce(&arg3);
  if(arg4.argtype == OP_ARG_GBL)
    global_reduce(&arg4);

  //update timer record
  op_timers_core(&cpu_t2, &wall_t2);
#ifdef COMM_PERF
  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  if(sent[0] == 1)op_mpi_perf_comm(k_i, arg0);
  if(sent[1] == 1)op_mpi_perf_comm(k_i, arg1);
  if(sent[2] == 1)op_mpi_perf_comm(k_i, arg2);
  if(sent[3] == 1)op_mpi_perf_comm(k_i, arg3);
  if(sent[4] == 1)op_mpi_perf_comm(k_i, arg4);
#endif
}

/*******************************************************************************
* op_par_loop template for 6 arguments
*******************************************************************************/

template < class T0, class T1, class T2, class T3,
           class T4, class T5 >
void op_par_loop(void (*kernel)( T0*, T1*, T2*, T3*,
                                 T4*, T5* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5 )
{
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4, *p_arg5;
  int exec_length = 0;

  int sent[6] = {0,0,0,0,0,0};

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if(arg0.idx != -1 || arg1.idx != -1 || arg2.idx != -1 || arg3.idx != -1 ||
      arg4.idx != -1 || arg5.idx != -1)//indirect loop
  {
    if (OP_diags==1) {
      if(arg0.argtype == OP_ARG_DAT) reset_halo(arg0);
      if(arg1.argtype == OP_ARG_DAT) reset_halo(arg1);
      if(arg2.argtype == OP_ARG_DAT) reset_halo(arg2);
      if(arg3.argtype == OP_ARG_DAT) reset_halo(arg3);
      if(arg4.argtype == OP_ARG_DAT) reset_halo(arg4);
      if(arg5.argtype == OP_ARG_DAT) reset_halo(arg5);
    }

    //for each indirect data set
    if(arg0.argtype == OP_ARG_DAT) sent[0] = exchange_halo(arg0);
    if(arg1.argtype == OP_ARG_DAT) sent[1] = exchange_halo(arg1);
    if(arg2.argtype == OP_ARG_DAT) sent[2] = exchange_halo(arg2);
    if(arg3.argtype == OP_ARG_DAT) sent[3] = exchange_halo(arg3);
    if(arg4.argtype == OP_ARG_DAT) sent[4] = exchange_halo(arg4);
    if(arg5.argtype == OP_ARG_DAT) sent[5] = exchange_halo(arg5);
  }

  //if all indirect dataset access is with OP_READ
  exec_length = set->size;
  //else
  if (arg0.idx != -1 && arg0.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg1.idx != -1 && arg1.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg2.idx != -1 && arg2.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg3.idx != -1 && arg3.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg4.idx != -1 && arg4.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg5.idx != -1 && arg5.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;

  // loop over set elements
  //(1) over core partition
  for (int n=0; n<core_num[set->index]; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );
    op_arg_set(n,arg5 ,&p_arg5 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5 );
  }

  //wait for comms to complete
  if(arg0.argtype == OP_ARG_DAT) if(sent[0] == 1 )wait_all(arg0);
  if(arg1.argtype == OP_ARG_DAT) if(sent[1] == 1 )wait_all(arg1);
  if(arg2.argtype == OP_ARG_DAT) if(sent[2] == 1 )wait_all(arg2);
  if(arg3.argtype == OP_ARG_DAT) if(sent[3] == 1 )wait_all(arg3);
  if(arg4.argtype == OP_ARG_DAT) if(sent[4] == 1 )wait_all(arg4);
  if(arg5.argtype == OP_ARG_DAT) if(sent[5] == 1 )wait_all(arg5);

  for (int n=core_num[set->index]; n<set->size; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );
    op_arg_set(n,arg5 ,&p_arg5 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5);
  }

  //(2) over exec halo (blank out global parameters to avoid double counting)
  for (int n=set->size; n<exec_length; n++) {
    op_arg_set(n,*(blank_arg(&arg0)),&p_arg0 );
    op_arg_set(n,*(blank_arg(&arg1)),&p_arg1 );
    op_arg_set(n,*(blank_arg(&arg2)),&p_arg2 );
    op_arg_set(n,*(blank_arg(&arg3)),&p_arg3 );
    op_arg_set(n,*(blank_arg(&arg4)),&p_arg4 );
    op_arg_set(n,*(blank_arg(&arg5)),&p_arg5 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5);
  }

  //set dirty bit on direct/indirect datasets with access OP_INC,OP_WRITE, OP_RW
  if(arg0.argtype == OP_ARG_DAT)set_dirtybit(arg0);
  if(arg1.argtype == OP_ARG_DAT)set_dirtybit(arg1);
  if(arg2.argtype == OP_ARG_DAT)set_dirtybit(arg2);
  if(arg3.argtype == OP_ARG_DAT)set_dirtybit(arg3);
  if(arg4.argtype == OP_ARG_DAT)set_dirtybit(arg4);
  if(arg5.argtype == OP_ARG_DAT)set_dirtybit(arg5);

  //performe any global operations
  if(arg0.argtype == OP_ARG_GBL)
    global_reduce(&arg0);
  if(arg1.argtype == OP_ARG_GBL)
    global_reduce(&arg1);;
  if(arg2.argtype == OP_ARG_GBL)
    global_reduce(&arg2);
  if(arg3.argtype == OP_ARG_GBL)
    global_reduce(&arg3);
  if(arg4.argtype == OP_ARG_GBL)
    global_reduce(&arg4);
  if(arg5.argtype == OP_ARG_GBL)
    global_reduce(&arg5);

  //update timer record
  op_timers_core(&cpu_t2, &wall_t2);

  //update performance records
#ifdef COMM_PERF
  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  if(sent[0] == 1)op_mpi_perf_comm(k_i, arg0);
  if(sent[1] == 1)op_mpi_perf_comm(k_i, arg1);
  if(sent[2] == 1)op_mpi_perf_comm(k_i, arg2);
  if(sent[3] == 1)op_mpi_perf_comm(k_i, arg3);
  if(sent[4] == 1)op_mpi_perf_comm(k_i, arg4);
  if(sent[5] == 1)op_mpi_perf_comm(k_i, arg5);
#endif
}

/*******************************************************************************
* op_par_loop template for 8 arguments
*******************************************************************************/

template < class T0, class T1, class T2, class T3,
           class T4, class T5, class T6, class T7 >
void op_par_loop(void (*kernel)( T0*, T1*, T2*, T3*,
                                 T4*, T5*, T6*, T7* ),
  char const * name, op_set set,
  op_arg arg0, op_arg arg1, op_arg arg2, op_arg arg3,
  op_arg arg4, op_arg arg5, op_arg arg6, op_arg arg7 )
{
  char *p_arg0, *p_arg1, *p_arg2, *p_arg3,
       *p_arg4, *p_arg5, *p_arg6, *p_arg7;
  int exec_length = 0;

  int sent[8] = {0,0,0,0,0,0,0,0};

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
    op_arg_check(set,0 ,arg0 ,&ninds,name);
    op_arg_check(set,1 ,arg1 ,&ninds,name);
    op_arg_check(set,2 ,arg2 ,&ninds,name);
    op_arg_check(set,3 ,arg3 ,&ninds,name);
    op_arg_check(set,4 ,arg4 ,&ninds,name);
    op_arg_check(set,5 ,arg5 ,&ninds,name);
    op_arg_check(set,6 ,arg6 ,&ninds,name);
    op_arg_check(set,7 ,arg7 ,&ninds,name);
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %s \n",name);
    else
      printf(" kernel routine with indirection: %s \n",name);
  }

  // initialise timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers_core(&cpu_t1, &wall_t1);

  if(arg0.idx != -1 || arg1.idx != -1 || arg2.idx != -1 || arg3.idx != -1 ||
      arg4.idx != -1 || arg5.idx != -1 || arg6.idx != -1 || arg7.idx != -1)//indirect loop
  {
    if (OP_diags==1) {
      if(arg0.argtype == OP_ARG_DAT) reset_halo(arg0);
      if(arg1.argtype == OP_ARG_DAT) reset_halo(arg1);
      if(arg2.argtype == OP_ARG_DAT) reset_halo(arg2);
      if(arg3.argtype == OP_ARG_DAT) reset_halo(arg3);
      if(arg4.argtype == OP_ARG_DAT) reset_halo(arg4);
      if(arg5.argtype == OP_ARG_DAT) reset_halo(arg5);
      if(arg6.argtype == OP_ARG_DAT) reset_halo(arg6);
      if(arg7.argtype == OP_ARG_DAT) reset_halo(arg7);
    }

    //for each indirect data set
    if(arg0.argtype == OP_ARG_DAT) sent[0] = exchange_halo(arg0);
    if(arg1.argtype == OP_ARG_DAT) sent[1] = exchange_halo(arg1);
    if(arg2.argtype == OP_ARG_DAT) sent[2] = exchange_halo(arg2);
    if(arg3.argtype == OP_ARG_DAT) sent[3] = exchange_halo(arg3);
    if(arg4.argtype == OP_ARG_DAT) sent[4] = exchange_halo(arg4);
    if(arg5.argtype == OP_ARG_DAT) sent[5] = exchange_halo(arg5);
    if(arg6.argtype == OP_ARG_DAT) sent[6] = exchange_halo(arg6);
    if(arg7.argtype == OP_ARG_DAT) sent[7] = exchange_halo(arg7);
  }

  //for all indirect dataset access with OP_READ
  exec_length = set->size;

  if (arg0.idx != -1 && arg0.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg1.idx != -1 && arg1.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg2.idx != -1 && arg2.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg3.idx != -1 && arg3.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg4.idx != -1 && arg4.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg5.idx != -1 && arg5.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg6.idx != -1 && arg6.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;
  else if (arg7.idx != -1 && arg7.acc != OP_READ)
    exec_length = set->size + OP_import_exec_list[set->index]->size;

  // loop over set elements
  //(1) over core partition
  for (int n=0; n<core_num[set->index]; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );
    op_arg_set(n,arg5 ,&p_arg5 );
    op_arg_set(n,arg6 ,&p_arg6 );
    op_arg_set(n,arg7 ,&p_arg7 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5,  (T6 *)p_arg6,  (T7 *)p_arg7 );
  }

  //wait for comms to complete
  if(arg0.argtype == OP_ARG_DAT && sent[0] == 1 )wait_all(arg0);
  if(arg1.argtype == OP_ARG_DAT && sent[1] == 1 )wait_all(arg1);
  if(arg2.argtype == OP_ARG_DAT && sent[2] == 1 )wait_all(arg2);
  if(arg3.argtype == OP_ARG_DAT && sent[3] == 1 )wait_all(arg3);
  if(arg4.argtype == OP_ARG_DAT && sent[4] == 1 )wait_all(arg4);
  if(arg5.argtype == OP_ARG_DAT && sent[5] == 1 )wait_all(arg5);
  if(arg6.argtype == OP_ARG_DAT && sent[6] == 1 )wait_all(arg6);
  if(arg7.argtype == OP_ARG_DAT && sent[7] == 1 )wait_all(arg7);

  for (int n=core_num[set->index]; n<set->size; n++) {
    op_arg_set(n,arg0 ,&p_arg0 );
    op_arg_set(n,arg1 ,&p_arg1 );
    op_arg_set(n,arg2 ,&p_arg2 );
    op_arg_set(n,arg3 ,&p_arg3 );
    op_arg_set(n,arg4 ,&p_arg4 );
    op_arg_set(n,arg5 ,&p_arg5 );
    op_arg_set(n,arg6 ,&p_arg6 );
    op_arg_set(n,arg7 ,&p_arg7 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5,  (T6 *)p_arg6,  (T7 *)p_arg7 );
  }

  //(2) over exec halo (blank out global parameters to avoid double counting)
  for (int n=set->size; n<exec_length; n++) {
    op_arg_set(n,*(blank_arg(&arg0)),&p_arg0 );
    op_arg_set(n,*(blank_arg(&arg1)),&p_arg1 );
    op_arg_set(n,*(blank_arg(&arg2)),&p_arg2 );
    op_arg_set(n,*(blank_arg(&arg3)),&p_arg3 );
    op_arg_set(n,*(blank_arg(&arg4)),&p_arg4 );
    op_arg_set(n,*(blank_arg(&arg5)),&p_arg5 );
    op_arg_set(n,*(blank_arg(&arg6)),&p_arg6 );
    op_arg_set(n,*(blank_arg(&arg7)),&p_arg7 );

    // call kernel function, passing in pointers to data
    kernel( (T0 *)p_arg0,  (T1 *)p_arg1,  (T2 *)p_arg2,  (T3 *)p_arg3,
        (T4 *)p_arg4,  (T5 *)p_arg5,  (T6 *)p_arg6,  (T7 *)p_arg7 );
  }

  //set dirty bit on direct/indirect datasets with access OP_INC,OP_WRITE, OP_RW
  if(arg0.argtype == OP_ARG_DAT)set_dirtybit(arg0);
  if(arg1.argtype == OP_ARG_DAT)set_dirtybit(arg1);
  if(arg2.argtype == OP_ARG_DAT)set_dirtybit(arg2);
  if(arg3.argtype == OP_ARG_DAT)set_dirtybit(arg3);
  if(arg4.argtype == OP_ARG_DAT)set_dirtybit(arg4);
  if(arg5.argtype == OP_ARG_DAT)set_dirtybit(arg5);
  if(arg6.argtype == OP_ARG_DAT)set_dirtybit(arg6);
  if(arg7.argtype == OP_ARG_DAT)set_dirtybit(arg7);

  //performe any global operations
  if(arg0.argtype == OP_ARG_GBL)
    global_reduce(&arg0);
  if(arg1.argtype == OP_ARG_GBL)
    global_reduce(&arg1);;
  if(arg2.argtype == OP_ARG_GBL)
    global_reduce(&arg2);
  if(arg3.argtype == OP_ARG_GBL)
    global_reduce(&arg3);
  if(arg4.argtype == OP_ARG_GBL)
    global_reduce(&arg4);
  if(arg5.argtype == OP_ARG_GBL)
    global_reduce(&arg5);
  if(arg6.argtype == OP_ARG_GBL)
    global_reduce(&arg6);
  if(arg7.argtype == OP_ARG_GBL)
    global_reduce(&arg7);

  //update timer record
  op_timers_core(&cpu_t2, &wall_t2);

  //update performance records
#ifdef COMM_PERF
  int k_i = op_mpi_perf_time(name, wall_t2 - wall_t1);
  if(sent[0] == 1)op_mpi_perf_comm(k_i, arg0);
  if(sent[1] == 1)op_mpi_perf_comm(k_i, arg1);
  if(sent[2] == 1)op_mpi_perf_comm(k_i, arg2);
  if(sent[3] == 1)op_mpi_perf_comm(k_i, arg3);
  if(sent[4] == 1)op_mpi_perf_comm(k_i, arg4);
  if(sent[5] == 1)op_mpi_perf_comm(k_i, arg5);
  if(sent[6] == 1)op_mpi_perf_comm(k_i, arg6);
  if(sent[7] == 1)op_mpi_perf_comm(k_i, arg7);
#endif
}

#endif /* __OP_MPI_SEQ_H */

