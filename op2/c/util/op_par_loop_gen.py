#! /usr/bin/env python

import os, sys
maxargs = int(sys.argv[1]) if len(sys.argv) > 1 else 8

file_h = os.path.join(os.path.abspath(os.path.dirname(__file__)),"../include/op_seq_mat.h")

header_h = """/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009, Mike Giles
* Copyright (c) 2011, Florian Rathgeber
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

#ifndef __OP_SEQ_H
#define __OP_SEQ_H

#include "op_lib_mat.h"
#include <boost/type_traits.hpp>

static inline void op_arg_set(int n, op_arg arg, char **p_arg){
  int n2;
  if (arg.map==NULL)         // identity mapping, or global data
    n2 = n;
  else                       // standard pointers
    n2 = arg.map->map[arg.idx+n*arg.map->dim];
  *p_arg = arg.data + n2*arg.size;
}

static inline void copy_in(int n, op_arg arg, char **p_arg) {
  // For each index in the target dimension of the map, copy the pointer to
  // the data
  for (int i = 0; i < arg.map->dim; ++i)
      p_arg[i] = arg.data + arg.map->map[i+n*arg.map->dim]*arg.size;
}

op_itspace op_iteration_space(op_set set, int i, int j)
{
  op_itspace ret = (op_itspace)malloc(sizeof(op_itspace_core));
  ret->set = set;
  ret->ndims = 2;
  ret->dims = (int *)malloc(ret->ndims * sizeof(int));
  ret->dims[0] = i;
  ret->dims[1] = j;
  return ret;
}

op_itspace op_iteration_space(op_set set)
{
  op_itspace ret = (op_itspace)malloc(sizeof(op_itspace_core));
  ret->set = set;
  ret->ndims = 0;
  ret->dims = NULL;
  return ret;
}

op_itspace op_iteration_space(op_set set, int i)
{
  op_itspace ret = (op_itspace)malloc(sizeof(op_itspace_core));
  ret->set = set;
  ret->ndims = 1;
  ret->dims = (int *)malloc(ret->ndims * sizeof(int));
  ret->dims[0] = i;
  return ret;
}
"""

footer_h = """

#endif
"""

templates = {

    'par_loop_comment': """
//
// op_par_loop routine for %d arguments
//

""",

    'par_loop_body': """
{
%(argdefs)s

  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
%(argchecks)s
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %%s \\n",name);
    else
      printf(" kernel routine with indirection: %%s \\n",name);
  }

  // Allocate memory for vector map indices
%(allocate)s
  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices
%(argsetters)s
    // call kernel function, passing in pointers to data
%(kernelcall)s
    // Assemble local matrix into global matrix
%(mataddto)s
  }

  // Free memory for vector map indices
%(free)s
  // Global matrix assembly
%(matassembly)s
}
""",

    'allocate': """
  switch ( arg%d.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg%d = arg%d.data;
      break;
    case OP_ARG_DAT:
      if (arg%d.idx  < -1)
        p_arg%d = (char *)malloc(arg%d.map->dim*sizeof(T%d));
      break;
    case OP_ARG_MAT:
      p_arg%d = (char*) malloc(arg%d.map->dim * arg%d.map2->dim * arg%d.size);
      break;
  }
""",

    'argsetters': """
    if (arg%d.argtype == OP_ARG_DAT) {
      if (arg%d.idx < -1)
        copy_in(n, arg%d, (char**)p_arg%d);
      else
        op_arg_set(n, arg%d, &p_arg%d );
    }
""",

    'mataddto': """
    if (arg%d.argtype == OP_ARG_MAT) {
      const int rows = arg%d.map->dim;
      const int cols = arg%d.map2->dim;
      op_mat_addto( arg%d.mat, p_arg%d, rows, arg%d.map->map + n*rows, cols, arg%d.map2->map + n*cols);
    }
""",

    'free': """
  if ((arg%d.argtype == OP_ARG_DAT && arg%d.idx < -1) || arg%d.argtype == OP_ARG_MAT) free(p_arg%d);
""",

    'par_loop_itspace_comment': """
//
// op_par_loop routine for %d arguments with op_iteration_space call
//

""",
    'par_loop_itspace_body': """
{
%(argdefs)s
  op_set set = itspace->set;
  // consistency checks

  int ninds=0;

  if (OP_diags>0) {
%(argchecks)s
  }

  if (OP_diags>2) {
    if (ninds==0)
      printf(" kernel routine w/o indirection:  %%s \\n",name);
    else
      printf(" kernel routine with indirection: %%s \\n",name);
  }

  // Allocate memory for vector map indices
%(allocate_itspace)s
  // loop over set elements

  for (int n=0; n<set->size; n++) {
    // Copy in of vector map indices
%(argsetters)s
    // call kernel function, passing in pointers to data
    int ilower = 0;
    int iupper = itspace->dims[0];
    int jlower = 0;
    int jupper = itspace->dims[1];
    int idxs[2];
%(itspace_loop_prelim)s
%(itspace_loop)s
%(itspace_zero_mat)s
%(kernelcall_itspace)s
        // Assemble local matrix into global matrix
%(mataddto_itspace)s
      }
    }
  }

  // Free memory for vector map indices
%(free)s
  free(itspace->dims);
  free(itspace);
  // Global matrix assembly
%(matassembly)s
}
""",
    'allocate_itspace': """
  switch ( arg%d.argtype ) {
    // Globals need their pointer only set once before the loop
    case OP_ARG_GBL:
      p_arg%d = arg%d.data;
      break;
    case OP_ARG_DAT:
      if (arg%d.idx < -1)
        p_arg%d = (char *)malloc(arg%d.map->dim*sizeof(T%d));
      break;
    case OP_ARG_MAT:
      p_arg%d = (char*) malloc(sizeof(T%d));
      break;
  }
""",
    'itspace_loop_prelim' : """
    int arg%didxs[2] = {0, 1};
    if (arg%d.argtype == OP_ARG_MAT) {
      int iut;
      int jut;
      if (arg%d.idx < OP_I_OFFSET) {
        iut = itspace->dims[op_i(arg%d.idx)-1];
        arg%didxs[0] = op_i(arg%d.idx) - 1;
      } else if (arg%d.idx < -1) {
        iut = arg%d.map->dim;
      } else {
        printf("Invalid index (not vector index or op_i) for arg %d, aborting\\n");
        exit(-1);
      }
      if (arg%d.idx2 < OP_I_OFFSET) {
        jut = itspace->dims[op_i(arg%d.idx2)-1];
        arg%didxs[1] = op_i(arg%d.idx2) - 1;
      } else if (arg%d.idx2 < -1) {
        jut = arg%d.map2->dim;
      } else {
        printf("Invalid index (not vector index or op_i) for arg %d, aborting\\n");
        exit(-1);
      }
      if (iut != iupper || jut != jupper) {
        printf("Map dimensions do not match iteration space, aborting\\n");
        exit(-1);
      }
    }
""",
    'itspace_loop' : """
    for (idxs[0] = ilower; idxs[0] < iupper; idxs[0]++) {
      for (idxs[1] = jlower; idxs[1] < jupper; idxs[1]++ ) {
""",
    'itspace_zero_mat' : """
        if (arg%d.argtype == OP_ARG_MAT) {
          ((T%d *)p_arg%d)[0] = (T%d)0;
        }
""",
    'mataddto_itspace' : """
        if (arg%d.argtype == OP_ARG_MAT) {
          const int rows = arg%d.map->dim;
          const int cols = arg%d.map2->dim;
          op_mat_addto_scalar(arg%d.mat, p_arg%d,
                              arg%d.map->map[n*rows + idxs[arg%didxs[0]]],
                              arg%d.map2->map[n*cols + idxs[arg%didxs[1]]]);
        }
"""
    }

def format_block(head, tail, body, sep, b, n, k):
    return head \
        + (',\n'+' '*len(head)).join( \
        [sep.join( \
                [(body % ((i,)*b)) for i in range(s,min(n,s+k))] \
                    ) for s in range(0,n,k)] \
            ) \
            + tail

with open(file_h,"w") as h:

    h.write(header_h)

    # Loop over arguments
    for n in range(1,maxargs+1):
        # build op_par_loop signature
        par_loop_comment = templates['par_loop_comment'] % n
        par_loop_sig  = format_block("template < "                        , " >\n" , "class T%d", ", ", 1, n, 4)
        par_loop_sig += format_block("void op_par_loop ( void (*kernel)( ", " ),\n", "T%d*"     , ", ", 1, n, 4)
        par_loop_sig += "  char const * name, op_set set,\n"
        par_loop_sig += format_block("  ", " )" , "op_arg arg%d", ", ", 1, n, 4)
        par_loop_body = templates['par_loop_body'] % {
            'argdefs': format_block("  char ", ";", "*p_arg%d = 0", ", ", 1, n, 4),
            'argchecks': '\n'.join(["    op_arg_check(set,%d ,arg%d ,&ninds,name);" % (i,i) for i in range(n)]),
            'allocate': ''.join([templates['allocate'] % ((i,)*11) for i in range(n)]),
            'argsetters': ''.join([templates['argsetters'] % ((i,)*6) for i in range(n)]),
            'mataddto': ''.join([templates['mataddto'] % ((i,)*7) for i in range(n)]),
            'kernelcall': format_block("    kernel( ", " );", "(T%d *)p_arg%d", ", ", 2, n, 4),
            'free': ''.join([templates['free'] % ((i,)*4) for i in range(n)]),
            'matassembly': '\n'.join(["  if (arg%d.argtype == OP_ARG_MAT) op_mat_assemble(arg%d.mat);" % ((i,)*2) for i in range(n)])
            }
        h.write(par_loop_comment + par_loop_sig + par_loop_body)
        par_loop_comment = templates['par_loop_itspace_comment'] % n
        par_loop_sig  = format_block("template < "                        , " >\n" , "class T%d", ", ", 1, n, 4)
        par_loop_sig += format_block("void op_par_loop ( void (*kernel)( ",
                                     ", int, int ),\n",
                                     "T%d*"     , ", ", 1, n, 4)
        par_loop_sig += "  char const * name, op_itspace itspace,\n"
        par_loop_sig += format_block("  ", " )" , "op_arg arg%d", ", ", 1, n, 4)
        par_loop_body = templates['par_loop_itspace_body'] % {
            'argdefs': format_block("  char ", ";", "*p_arg%d = 0", ", ", 1, n, 4),
            'argchecks': '\n'.join(["    op_arg_check(set,%d ,arg%d ,&ninds,name);" % (i,i) for i in range(n)]),
            'allocate_itspace': ''.join([templates['allocate_itspace'] % ((i,)*9) for i in range(n)]),
            'argsetters': ''.join([templates['argsetters'] % ((i,)*6) for i in range(n)]),
            'itspace_loop_prelim' : ''.join([templates['itspace_loop_prelim'] % ((i,)*16) for i in range(n)]),
            'itspace_loop' : ''.join([templates['itspace_loop']]),
            'itspace_zero_mat' : ''.join([templates['itspace_zero_mat'] % ((i,)*4) for i in range(n)]),
            'mataddto_itspace': ''.join([templates['mataddto_itspace'] % ((i,)*9) for i in range(n)]),
            'kernelcall_itspace': format_block("        kernel( ", ", idxs[0], idxs[1]);", "(T%d *)p_arg%d", ", ", 2, n, 4),
            'free': ''.join([templates['free'] % ((i,)*4) for i in range(n)]),
            'matassembly': '\n'.join(["  if (arg%d.argtype == OP_ARG_MAT) op_mat_assemble(arg%d.mat);" % ((i,)*2) for i in range(n)])
            }
        h.write(par_loop_comment + par_loop_sig + par_loop_body)
    h.write(footer_h)
