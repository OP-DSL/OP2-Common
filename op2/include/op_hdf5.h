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

#ifndef __OP_HDF5_H
#define __OP_HDF5_H

/*
 * op_hdf5.h
 *
 * Header file for the parallel I/O functions
 *
 * written by: Gihan R. Mudalige, (Started 10-10-2011)
 */

#ifdef __cplusplus
extern "C" {
#endif

op_set op_decl_set_hdf5(char const *file, char const *name);
op_set op_decl_set_hdf5_infer_size(char const *file, char const *name, char const* set_dataset_name);
op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file,
                        char const *name);
op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file,
                        char const *name);

void op_get_const_hdf5(char const *name, int dim, char const *type,
                       char *const_data, char const *file_name);

void op_dump_to_hdf5(char const *file_name);
void op_write_const_hdf5(char const *name, int dim, char const *type,
                         char *const_data, char const *file_name);

void op_fetch_data_hdf5_file(op_dat dat, char const *file_name);
void op_fetch_data_hdf5_file_path(op_dat dat, char const *file_name,
                                  char const *path_name);

#ifdef __cplusplus
}
#endif

#endif /* __OP_HDF5_H */
