/*  Open source copyright declaration based on BSD open source template:
 *  http://www.opensource.org/licenses/bsd-license.php
 *
 * Copyright (c) 2011-2012, Carlo Bertolli, Florian Rathgeber
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdlib.h>
#include <op_lib_core.h>
#include <op_hdf5.h>

// Used in all those backends without HDF5
op_set op_decl_set_hdf5(char const *file, char const *name) {
  (void)file; (void)name;
  return NULL;
}
op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file, char const *name) {
  (void)from; (void) to; (void)dim; (void)file; (void)name;
  return NULL;
}
op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file, char const *name) {
  (void)set; (void)dim; (void)type; (void)file; (void)name;
  return NULL;
}
void op_get_const_hdf5(char const *name, int dim, char const *type, char* const_data,
  char const *file_name) {(void)name; (void)dim; (void)type; (void)const_data; (void)file_name;}
void op_write_hdf5(char const * file_name) { (void) file_name;}
void op_write_const_hdf5(char const *name, int dim, char const *type, char* const_data,
  char const *file_name) { (void)name; (void)dim; (void)type; (void)const_data; (void)file_name; }
