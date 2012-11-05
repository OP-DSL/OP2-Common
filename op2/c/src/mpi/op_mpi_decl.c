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

/*
 * This file implements the user-level OP2 functions for the case
 * of the mpi back-end
 */

#include <mpi.h>
#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_mpi_core.h>
#include <petsc.h>
/*
 * Routines called by user code and kernels
 * these wrappers are used by non-CUDA versions
 * op_lib.cu provides wrappers for CUDA version
 */

extern MPI_Comm OP_MPI_WORLD;
void op_init ( int argc, char ** argv, int diags )
{
  int flag = 0;
  MPI_Initialized(&flag);
  if(!flag)
  {
    MPI_Init(&argc, &argv);
  }
  MPI_Comm_dup(MPI_COMM_WORLD, &OP_MPI_WORLD);
  PETSC_COMM_WORLD = OP_MPI_WORLD;
  PetscInitialize(&argc,&argv,(char *)0,(char *)0);

  op_init_core ( argc, argv, diags );
}

op_dat op_decl_dat_char( op_set set, int dim, char const * type, int size, char * data, char const *name )
{
  char* d = (char*) malloc(set->size*dim*size);
  if (d == NULL) {
    printf ( " op_decl_dat_char error -- error allocating memory to dat\n" );
    exit ( -1 );
  }
  memcpy(d, data, set->size*dim*size*sizeof(char));
  op_dat out_dat = op_decl_dat_core ( set, dim, type, size, d, name );
  out_dat-> user_managed = 0;
  return out_dat;


}

op_dat op_decl_dat_temp_char(op_set set, int dim, char const * type, int size, char const *name )
{
  char* d = NULL;
  op_dat dat = op_decl_dat_temp_core ( set, dim, type, size, d, name );

  //create empty data block to assign to this temporary dat (including the halos)
  int halo_size = OP_import_exec_list[set->index]->size +
                  OP_import_nonexec_list[set->index]->size;

  dat->data = (char*) calloc((set->size+halo_size)*dim*size, 1); //initialize data bits to 0
  dat-> user_managed = 0;

  //need to allocate mpi_buffers for this new temp_dat
  op_mpi_buffer mpi_buf= (op_mpi_buffer)xmalloc(sizeof(op_mpi_buffer_core));

  halo_list exec_e_list = OP_export_exec_list[set->index];
  halo_list nonexec_e_list = OP_export_nonexec_list[set->index];

  mpi_buf->buf_exec = (char *)xmalloc((exec_e_list->size)*dat->size);
  mpi_buf->buf_nonexec = (char *)xmalloc((nonexec_e_list->size)*dat->size);

  halo_list exec_i_list = OP_import_exec_list[set->index];
  halo_list nonexec_i_list = OP_import_nonexec_list[set->index];

  mpi_buf->s_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
      (exec_e_list->ranks_size + nonexec_e_list->ranks_size));
  mpi_buf->r_req = (MPI_Request *)xmalloc(sizeof(MPI_Request)*
      (exec_i_list->ranks_size + nonexec_i_list->ranks_size));

  mpi_buf->s_num_req = 0;
  mpi_buf->r_num_req = 0;

  dat->mpi_buffer = mpi_buf;

  return dat;
}

int op_free_dat_temp_char ( op_dat dat )
{
  //need to free mpi_buffers used in this op_dat
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_exec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->buf_nonexec);
  free(((op_mpi_buffer)(dat->mpi_buffer))->s_req);
  free(((op_mpi_buffer)(dat->mpi_buffer))->r_req);
  free(dat->mpi_buffer);
  return op_free_dat_temp_core (dat);
}

void op_fetch_data ( op_dat dat )
{
  (void)dat;
}

/*
 * No specific action is required for constants in MPI
 */

void op_decl_const_char ( int dim, char const * type, int typeSize, char * data, char const * name )
{
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

op_plan *
op_plan_get ( char const * name, op_set set, int part_size,
              int nargs, op_arg * args, int ninds, int *inds )
{
  return op_plan_core ( name, set, part_size, nargs, args, ninds, inds );
}

void op_printf(const char* format, ...)
{
  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  if(my_rank==MPI_ROOT)
  {
    va_list argptr;
    va_start(argptr, format);
    vprintf(format, argptr);
    va_end(argptr);
  }
}

void op_exit()
{
  op_mpi_exit();
  op_rt_exit();
  op_exit_core();

  int flag = 0;
  MPI_Finalized(&flag);
  if(!flag)
    MPI_Finalize();
}

/*
 * Wrappers of core lib
 */

op_set op_decl_set(int size, char const *name )
{
  return op_decl_set_core ( size, name );
}

op_map op_decl_map(op_set from, op_set to, int dim, int * imap, char const * name )
{
  int* m = (int*) malloc(from->size*dim*sizeof(int));
  memcpy(m, imap, from->size*dim*sizeof(int));

  op_map out_map= op_decl_map_core ( from, to, dim, m, name );
  out_map-> user_managed = 0;
  return out_map;
  //return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg op_arg_dat( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

op_arg
op_arg_gbl_char ( char * data, int dim, const char *type, int size, op_access acc )
{
  return op_arg_gbl_core ( data, dim, type, size, acc );
}

void op_timers(double * cpu, double * et)
{
  MPI_Barrier(MPI_COMM_WORLD);
  op_timers_core(cpu,et);
}


void op_timing_output()
{
   op_timing_output_core();
   mpi_timing_output();
}
