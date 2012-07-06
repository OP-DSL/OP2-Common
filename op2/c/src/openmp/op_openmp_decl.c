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
 * of the OpenMP back-end
 */

#include <op_lib_c.h>
#include <op_rt_support.h>

/*
 * Routines called by user code and kernels
 * these wrappers are used by non-CUDA versions
 * op_lib.cu provides wrappers for CUDA version
 */

void
op_init ( int argc, char ** argv, int diags )
{
  op_init_core ( argc, argv, diags );
}

op_dat
op_decl_dat_char ( op_set set, int dim, char const * type, int size, char * data, char const *name )
{
  return op_decl_dat_core ( set, dim, type, size, data, name );
}

void
op_fetch_data ( op_dat dat )
{
  (void)dat;
}


/*
 * No specific action is required for constants in OpenMP
 */

void
op_decl_const_char ( int dim, char const * type, int typeSize, char * data, char const * name )
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

void
op_exit (  )
{
  op_rt_exit (  );
  op_exit_core (  );
}

/*
 * Wrappers of core lib
 */

op_set
op_decl_set ( int size, char const *name )
{
  return op_decl_set_core ( size, name );
}

op_map
op_decl_map ( op_set from, op_set to, int dim, int * imap, char const * name )
{
  return op_decl_map_core ( from, to, dim, imap, name );
}

op_arg
op_arg_dat ( op_dat dat, int idx, op_map map, int dim, char const * type, op_access acc )
{
  return op_arg_dat_core ( dat, idx, map, dim, type, acc );
}

void op_timers(double * cpu, double * et)
{
  op_timers_core(cpu,et);
}

op_arg
op_arg_gbl_char ( char * data, int dim, const char *type, int size, op_access acc )
{
  return op_arg_gbl_core ( data, dim, type, size, acc );
}

void op_timing_output()
{
   op_timing_output_core();
}
