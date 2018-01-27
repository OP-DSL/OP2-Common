/*
* Open source copyright declaration based on BSD open source template:
* http://www.opensource.org/licenses/bsd-license.php
*
* This file is part of the OP2 distribution.
*
* Copyright (c) 2013, Mike Giles and others. Please see the AUTHORS file in
* the main source directory for a full list of copyright holders.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright
* notice, this list of conditions and the following disclaimer in the
* documentation and/or other materials provided with the distribution.
* * The name of Mike Giles may not be used to endorse or promote products
* derived from this software without specific prior written permission.
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

/** @brief OP core library functions - lazy execution
  * @author Gihan Mudalige, Istvan Reguly
  * @details Implementations of the core library functions utilized by all OPS
 *  backends. Specifically implements the lazy execution functionality
  */

#include "op_lib_core.h"

#include "op_hdf5.h"
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <float.h>
#ifndef DECIMAL_DIG
#define DECIMAL_DIG 17
#endif


static char *copy_str(char const *src) {
  const size_t len = strlen(src) + 1;
  char *dest = (char *)op_calloc(len, sizeof(char));
  return strncpy(dest, src, len);
}

extern "C" void op_generate_consts_header();

////////////////////////////////////////////////////////////////////////
// Global variables
/////////////////////////////////////////////////////////////////////////

bool consts_header_generated = false;

////////////////////////////////////////////////////////////////////////
// Enqueueing loops
// Execute loops given in the kernel descriptor
/////////////////////////////////////////////////////////////////////////

void op_enqueue_kernel(op_kernel_descriptor *desc) {

  if (!consts_header_generated) op_generate_consts_header();

  // op_printf("In op_enqueue_kernel ****\n");
  // If not tiling, have to do the halo exchanges here
  double t1, t2, c;
  if (OP_diags > 1)
    op_timers_core(&c, &t1);

  // Halo exchanges ?

  if (OP_diags > 1)
    op_timers_core(&c, &t2);

  // Run the kernel
  desc->function(desc);

  // Dirtybits ?
}

typedef struct op_const_descriptor {
  int dim;
  char const *type;
  int typeSize;
  char *data;
  char const *name;
} op_const_descriptor;
std::vector<op_const_descriptor> op_const_list(0);

void op_lazy_const(int dim, char const *type, int typeSize, char *data,
                        char const *name) {
  //Check if already added
  bool found = false;
  for (size_t i = 0; i < op_const_list.size(); i++) {
    if (!strcmp(op_const_list[i].name, name)) {
      found = true;
      if (strcmp(op_const_list[i].type, type) || op_const_list[i].dim != dim) {
        op_printf("Error, incompatible redefitiontion of constant '%s'!\n",op_const_list[i].name);
        exit(-1);
      } else {
        op_const_list[i].data = data;
      }
    }
  }

  if (!found) {
    op_const_list.push_back({dim, copy_str(type), typeSize, data, copy_str(name)});
  }
}


void op_generate_consts_header() {
  FILE *f = fopen("jit_const.h", "w"); // create only if file does not exist
  if (f == NULL) {
    printf("Error opening file!\n");
    exit(1);
  }

  for (size_t i = 0; i < op_const_list.size(); i++) {
    if (op_const_list[i].dim == 1) {
      if (!strcmp(op_const_list[i].type,"double")) {
        fprintf(f, "#define %s %.*e\n", op_const_list[i].name, DECIMAL_DIG, *((double*)op_const_list[i].data));
      } else if (!strcmp(op_const_list[i].type,"real(8)")) {
        fprintf(f, "#define %s %.*e_8\n", op_const_list[i].name, DECIMAL_DIG, *((double*)op_const_list[i].data));
      } else if (!strcmp(op_const_list[i].type,"float")) {
        fprintf(f, "#define %s %.*e\n", op_const_list[i].name, DECIMAL_DIG, *((float*)op_const_list[i].data));
      } else if (!strcmp(op_const_list[i].type,"real(4)")) {
        fprintf(f, "#define %s %.*e_4\n", op_const_list[i].name, DECIMAL_DIG, *((float*)op_const_list[i].data));
      } else if (!strcmp(op_const_list[i].type,"int") ||
          !strcmp(op_const_list[i].type,"integer(4)")) {
        fprintf(f, "#define %s %d\n", op_const_list[i].name, *((int*)op_const_list[i].data));
      } else if (!strcmp(op_const_list[i].type,"bool") ||
          !strcmp(op_const_list[i].type,"logical")) {
        if (*((char*)op_const_list[i].data))
          fprintf(f, "#define %s .true.\n", op_const_list[i].name);
        else
          fprintf(f, "#define %s .false.\n", op_const_list[i].name);
      }
    } else {
      // fprintf(f, "extern %s %s[%d];\n", op_const_list[i].type,
      //    op_const_list[i].name, op_const_list[i].dim);
    }
  }
  fclose(f);
  consts_header_generated = true;
}
