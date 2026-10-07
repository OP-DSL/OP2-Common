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

#ifndef __OP_MPI_CORE_H
#define __OP_MPI_CORE_H

/*
 * op_mpi_core.h
 *
 * Headder file for the OP2 Distributed memory (MPI) halo creation,
 * halo exchange and support utility routines/functions
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */
#ifndef OP_MPI_CORE_NOMPI
#include <mpi.h>

/** Define the root MPI process **/
#ifdef MPI_ROOT
#undef MPI_ROOT
#endif
#define MPI_ROOT 0

/** extern variables for halo creation and exchange**/
extern MPI_Comm OP_MPI_WORLD;
extern MPI_Comm OP_MPI_GLOBAL;

#endif /* OP_MPI_CORE_NOMPI */
// Structs that don't need the MPI include

/** external variables **/


// Structs and functions that use MPI definitions
#ifndef OP_MPI_CORE_NOMPI

template <typename T>
MPI_Datatype get_mpi_type() {
  if (std::is_same<T, long long>::value) {
    return MPI_LONG_LONG;
  } else if (std::is_same<T, int>::value) {
    return MPI_INT;
  } else if (std::is_same<T, idx_g_t>::value) {
    return MPI_LONG_LONG;
  } else {
    throw std::runtime_error("Unsupported type");
  }
  // #endif
}

template <typename T>
MPI_Datatype get_mpi_type(T*) {
  return get_mpi_type<T>();
}

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
* Utility function prototypes
*******************************************************************************/

void decl_partition(op_set set, idx_g_t *g_index, int *partition);

int is_onto_map(op_map map);

/*******************************************************************************
* Core MPI lib function prototypes
*******************************************************************************/

void op_halo_create();

void op_halo_permap_create();

void op_halo_destroy();

op_dat op_mpi_get_data(op_dat dat);

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high);

void mpi_timing_output();

void op_mpi_exit();

void print_dat_to_txtfile_mpi(op_dat dat, const char *file_name);

void print_dat_to_binfile_mpi(op_dat dat, const char *file_name);

/* Scatter user data from the original block partition into the current
 * (repartitioned) layout of dat. ptr holds local_size elements, which must
 * match this rank's original block-partitioned set size. */
void op_mpi_put_data(op_dat dat, void *ptr, size_t local_size);

void op_mpi_init(int argc, char **argv, int diags, MPI_Fint global,
                 MPI_Fint local);
void op_mpi_init_soa(int argc, char **argv, int diags, MPI_Fint global,
                     MPI_Fint local, int soa);

/* Defined in op_mpi_decl.c, may need to be put in a seperate headder file */
size_t op_mv_halo_device(op_set set, op_dat dat);

/* Defined in op_mpi_decl.c, may need to be put in a seperate headder file */
size_t op_mv_halo_list_device();

void partition(const char *lib_name, const char *lib_routine, op_set prime_set,
               op_map prime_map, op_dat data);

void op_move_to_device();

#ifdef __cplusplus
}
#endif

#endif /* OP_MPI_CORE_NOMPI */
#endif /* __OP_MPI_CORE_H */
