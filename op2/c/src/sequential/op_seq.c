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

#include <petscsys.h>

#include <op_lib_c.h>

/*
 * This file implements thw wrappers of core library routines for
 * the OP2 reference implementation. These functions are also used
 * by the Fortran OP2 reference implementation.
 */

void op_init ( int argc, char ** argv, int diags )
{
  PetscInitialize(&argc,&argv,(char *)0,(char *)0);
  op_init_core ( argc, argv, diags );
}

op_set op_decl_set ( int size, char const * name )
{
  return op_decl_set_core ( size, name );
}

op_map op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  return op_decl_map_core ( from, to, dim, imap, name );
}

op_dat op_decl_dat_char ( op_set set, int dim, char const * type, int size, char * data, char const * name )
{
  return op_decl_dat_core ( set, dim, type, size, data, name );
}

/*
 * The following function is empty for the reference implementation
 * and is not present in the core library: it is only needed
 * in OP2 main program to signal OP2 compilers which are the constant
 * names in the program
 */

void op_decl_const_char ( int dim, char const * type, int typeSize, char * data, char const * name ) {
  (void)dim;
  (void)type;
  (void)typeSize;
  (void)data;
  (void)name;
}

op_arg op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

op_arg
op_arg_gbl_char ( char * data, int dim, const char *type, int size, op_access acc )
{
  return op_arg_gbl_core ( data, dim, type, size, acc );
}

void op_fetch_data ( op_dat a ) {
  (void)a;
}


int op_get_size(op_set set)
{
  return set->size;
}

void op_printf(const char* format, ...)
{
  va_list argptr;
  va_start(argptr, format);
  vprintf(format, argptr);
  va_end(argptr);
}

void op_timers(double * cpu, double * et)
{
    op_timers_core(cpu,et);
}

void op_exit ()
{
  op_exit_core ();
  PetscFinalize();
}

void op_timing_output()
{
   op_timing_output_core();
}
