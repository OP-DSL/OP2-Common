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

#include "op_lib_core.h"

void op_arg_set(int n, op_arg arg, char **p_arg){
  int n2;
  if (arg.map==NULL)         // identity mapping, or global data
    n2 = n;
  else                       // standard pointers
    n2 = arg.map->map[arg.idx+n*arg.map->dim];

  *p_arg = arg.data + n2*arg.size;
}

"""

footer_h = """

#endif
"""

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
        par_loop_comment = """
//
// op_par_loop routine for 1 arguments
//

"""
        par_loop_sig  = format_block("template < "                        , " >\n" , "class T%d", ", ", 1, n, 4)
        par_loop_sig += format_block("void op_par_loop ( void (*kernel)( ", " ),\n", "T%d*"     , ", ", 1, n, 4)
        par_loop_sig += "  char const * name, op_set set,\n"
        par_loop_sig += format_block("  ", " )" , "op_arg arg%d", ", ", 1, n, 4)
        par_loop_body = """
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

  // loop over set elements

  for (int n=0; n<set->size; n++) {
%(argsetters)s

    // call kernel function, passing in pointers to data

%(kernelcall)s
  }
}
""" % {
    'argdefs': format_block("  char ", ";", "*p_arg%d", ", ", 1, n, 4),
    'argchecks': '\n'.join(["    op_arg_check(set,%d ,arg%d ,&ninds,name);" % (i,i) for i in range(n)]),
    'argsetters': '\n'.join(["    op_arg_set(n,arg%d ,&p_arg%d );" % (i,i) for i in range(n)]),
    'kernelcall': format_block("    kernel( ", " );", "(T%d *)p_arg%d", ", ", 2, n, 4)
    }

        h.write(par_loop_comment + par_loop_sig + par_loop_body)
    h.write(footer_h)
