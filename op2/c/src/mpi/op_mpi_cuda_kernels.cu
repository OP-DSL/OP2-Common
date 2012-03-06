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

#include <op_lib_mpi.h>

__global__ void export_halo_gather(int* list, char * dat, int copy_size, int elem_size, char * export_buffer)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if (id<copy_size) {
      for (int i =0;i<elem_size;i++) {
          export_buffer[id*elem_size+i]=dat[list[id]*elem_size+i];
      }
    }

}

void gather_data_to_buffer(op_arg arg, halo_list exp_exec_list, halo_list exp_nonexec_list)
{
    int threads = 192;
    int blocks = 1+((exp_exec_list->size-1)/192);
    export_halo_gather<<<blocks,threads>>>(export_exec_list_d[arg.dat->set->index],
      arg.data_d, exp_exec_list->size, arg.dat->size, arg.dat->buffer_d);

    int blocks2 = 1+((exp_nonexec_list->size-1)/192);
    export_halo_gather<<<blocks2,threads>>>(export_nonexec_list_d[arg.dat->set->index],
      arg.data_d, exp_nonexec_list->size, arg.dat->size, arg.dat->buffer_d+exp_exec_list->size*arg.dat->size);
}

