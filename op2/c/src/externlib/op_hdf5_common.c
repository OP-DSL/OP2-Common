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
 * op_hdf5_common.c
 *
 * Implements the common utility functions for both seq and mpi based
 * HDF5 I/O routines for the OP2 back end
 *
 * written by: Gihan R. Mudalige, (Started 27-10-2017)
 */

// hdf5 header
#include <hdf5.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <op_util.h> //just to include xmalloc routine

typedef struct {
  const char *type_str; // dataset type as string
  hsize_t size;         // dataset size (first dimension)
  hsize_t dim;          // element size (second dimension)
  size_t elem_bytes;    // element byte-size
} op_hdf5_dataset_properties;

/* Temporarily switch off HDF5 error handler*/
void H5error_off(H5E_auto_t old_func, void *old_client_data) {
  /* Save old error handler */
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL); // turn off HDF5's auto error reporting
}

/* Restore previous error handler .. report hdf5 error stack automatically*/
void H5error_on(H5E_auto_t old_func, void *old_client_data) {
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);
}

const char *op_hdf5_type_to_string(hid_t t) {
  char *text = NULL;
  if (H5Tequal(t, H5T_NATIVE_INT)) {
    text = (char *)malloc(4 * sizeof(char));
    strcpy(text, "int");
  } else if (H5Tequal(t, H5T_NATIVE_LONG)) {
    text = (char *)malloc(5 * sizeof(char));
    strcpy(text, "long");
  } else if (H5Tequal(t, H5T_NATIVE_LLONG)) {
    text = (char *)malloc(10 * sizeof(char));
    strcpy(text, "long long");
  } else if (H5Tequal(t, H5T_NATIVE_FLOAT)) {
    text = (char *)malloc(6 * sizeof(char));
    strcpy(text, "float");
  } else if (H5Tequal(t, H5T_NATIVE_DOUBLE)) {
    text = (char *)malloc(7 * sizeof(char));
    strcpy(text, "double");
  } else {
    text = (char *)malloc(13 * sizeof(char));
    strcpy(text, "UNRECOGNISED");
  }

  return (const char *)text;
}

herr_t get_dataset_properties(hid_t dset_id,
                              op_hdf5_dataset_properties *dset_props) {
  hid_t status;

  if (dset_props == NULL) {
    return -1;
  }

  // Get dimension and size:
  hid_t dataspace = H5Dget_space(dset_id);
  if (dataspace < 0) {
    return -1;
  }
  int ndims = H5Sget_simple_extent_ndims(dataspace);
  if (ndims == 0) {
    dset_props->size = 0;
    dset_props->dim = 0;
    H5Sclose(dataspace);
  } else {
    hsize_t dims[ndims];
    hsize_t maxdims[ndims];
    status = H5Sget_simple_extent_dims(dataspace, dims, maxdims);
    H5Sclose(dataspace);
    if (status < 0) {
      return -1;
    }
    dset_props->size = dims[0];
    dset_props->dim = (ndims > 1) ? dims[1] : 1;
  }

  // Get type information:
  hid_t t = H5Dget_type(dset_id);
  if (t < 0) {
    return -1;
  }
  dset_props->type_str = op_hdf5_type_to_string(t);
  if (H5Tequal(t, H5T_NATIVE_INT)) {
    dset_props->elem_bytes = sizeof(int);
  } else if (H5Tequal(t, H5T_NATIVE_LONG)) {
    dset_props->elem_bytes = sizeof(long);
  } else if (H5Tequal(t, H5T_NATIVE_LLONG)) {
    dset_props->elem_bytes = sizeof(long long);
  } else if (H5Tequal(t, H5T_NATIVE_FLOAT)) {
    dset_props->elem_bytes = sizeof(float);
  } else if (H5Tequal(t, H5T_NATIVE_DOUBLE)) {
    dset_props->elem_bytes = sizeof(double);
  } else {
    size_t name_len = H5Iget_name(dset_id, NULL, 0);
    char name[name_len];
    H5Iget_name(dset_id, name, name_len + 1);
    op_printf("Error: Do not recognise type of dataset '%s'\n", name);
    exit(2);
  }
  dset_props->elem_bytes *= dset_props->dim;

  return 0;
}

/*create path specified by map or dat ->name for a map or dataset
  within an HDF5 file*/
void create_path(const char *name, hid_t file_id) {

  hid_t group_id;
  herr_t status;
  H5E_auto_t old_func;
  void *old_client_data;

  char *path = (char *)name;
  char *ssc;
  int k = 0;
  int size = 50;
  int c = 0;
  ssc = strstr(path, "/");
  char *buffer = (char *)xmalloc(50 * sizeof(char));
  while (ssc) {
    k = strlen(path) - strlen(ssc);
    if (k > 0) {
      char result[30];
      strncpy(result, &path[0], k);
      result[k] = '\0';
      if (size <= c + k + 1) {
        size = 2 * (size + c + k + 1);
        buffer = (char *)xrealloc(buffer, 2 * size * sizeof(char));
      }
      sprintf(&buffer[c], "/%s", result);
      c += 1 + k;

      // Create a group named "/result" in the file.
      H5error_off(old_func, old_client_data);
      status = H5Gget_objinfo(file_id, buffer, 0, NULL);
      H5error_on(old_func, old_client_data);

      // printf("status %d, %s\n",status,buffer);
      if (status != 0) {
        group_id =
            H5Gcreate2(file_id, buffer, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Gclose(group_id);
      }
    }
    path = &path[strlen(path) - strlen(ssc) + 1];
    ssc = strstr(path, "/");
  }
  free(buffer);
}
