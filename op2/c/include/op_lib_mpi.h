/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009, Mike Giles
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
 * written by: Gihan R. Mudalige, 01-03-2011
 */

#ifndef __OP_LIB_MPI_H
#define __OP_LIB_MPI_H

#include <op_lib_core.h>
#include <op_lib_cpp.h>
#include "op_rt_support.h"
#include <op_mpi_core.h>
#include <op_mpi_part_core.h>
#include <op_hdf5.h>



/** extern variables for halo creation and exchange**/
extern MPI_Comm OP_MPI_WORLD;

extern halo_list *OP_export_exec_list;//EEH list
extern halo_list *OP_import_exec_list;//IEH list

extern halo_list *OP_import_nonexec_list;//INH list
extern halo_list *OP_export_nonexec_list;//ENH list 

extern int* dirtybit;
extern op_mpi_buffer *OP_mpi_buffer_list;
extern int *core_num; 

extern int OP_part_index;
extern part *OP_part_list;
extern int** orig_part_range;

#endif /* __OP_LIB_MPI_H */

