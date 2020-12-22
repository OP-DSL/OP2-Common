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
 * op_hdf5.c
 *
 * Implements the HDF5 based I/O routines for the OP2 single node
 * back end
 *
 * written by: Gihan R. Mudalige, (Started 10-10-2011)
 */

// hdf5 header
#include <hdf5.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2

#include <op_util.h> //just to include xmalloc routine

#include "op_hdf5_common.c"

/*******************************************************************************
* Routine to read an op_set from an hdf5 file
*******************************************************************************/

op_set op_decl_set_hdf5(char const *file, char const *name) {
  // HDF5 APIs definitions
  hid_t file_id; // file identifier
  hid_t dset_id; // dataset identifier

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_set_hdf5()\n",
              file);
    exit(2);
  }

  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

  int l_size = 0;
  // read data
  H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &l_size);

  H5Dclose(dset_id);
  H5Fclose(file_id);

  return op_decl_set(l_size, name);
}

op_set op_decl_set_hdf5_infer_size(char const *file, char const *name, char const *set_dataset_name) {
  // HDF5 APIs definitions
  hid_t file_id; // file identifier
  hid_t dset_id; // dataset identifier
  herr_t status;

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_set_hdf5()\n",
              file);
    exit(2);
  }

  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    op_printf("Could not obtain read access to file '%s'\n", file);
    exit(2);
  }

  status = H5Lexists(file_id, set_dataset_name, H5P_DEFAULT);
  if (status < 0) {
    op_printf("Dataset '%s' not found in file '%s'\n", set_dataset_name, file);
    exit(2);
  }

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, set_dataset_name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("Could not open dataset '%s' in file '%s'\n", set_dataset_name, file);
    H5Fclose(file_id);
    return NULL;
  }

  // Get size of dataset:
  op_hdf5_dataset_properties dset_props;
  status = get_dataset_properties(dset_id, &dset_props);
  if (status < 0) {
    op_printf("Could not access dataset '%s' in file '%s'\n", set_dataset_name, file);
    exit(2);
  }
  int l_size = dset_props.size;

  H5Dclose(dset_id);
  H5Fclose(file_id);

  free((char*)dset_props.type_str);

  return op_decl_set(l_size, name);
}

/*******************************************************************************
* Routine to read an op_map from an hdf5 file
*******************************************************************************/

op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file,
                        char const *name) {
  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  herr_t status;

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_map_hdf5()\n",
              file);
    exit(2);
  }
  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);

  /* Save old error handler */
  H5E_auto_t old_func;
  void *old_client_data;
  H5error_off(&old_func, &old_client_data);

  /*open data set*/
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("op_map with name : %s not found in file : %s \n", name, file);
    H5Fclose(file_id);
    return NULL;
  }

  op_hdf5_dataset_properties dset_props;
  status = get_dataset_properties(dset_id, &dset_props);
  if (status < 0) {
    op_printf("Could not get properties of dataset '%s' in file '%s'\n", name, file);
    exit(2);
  }

  int g_size = dset_props.size;
  // check if size is accurate
  if (from->size != g_size) {
    op_printf("map from set size %d in file %s and size %d do not match \n",
              g_size, file, from->size);
    exit(2);
  }

  int map_dim = dset_props.dim;
  if (map_dim != dim) {
    op_printf("map.dim %d in file %s and dim %d do not match\n", map_dim, file,
              dim);
    exit(2);
  }

  const char *typ = dset_props.type_str;

  // Restore previous error handler .. report hdf5 error stack automatically
  H5error_on(old_func, old_client_data);

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  // initialize data buffer and read data
  int *map;
  if (strcmp(typ, "int") == 0) {
    map = (int *)xmalloc(sizeof(int) * g_size * dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  } else if (strcmp(typ, "long") == 0) {
    map = (int *)xmalloc(sizeof(long) * g_size * dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  } else if (strcmp(typ, "long long") == 0) {
    map = (int *)xmalloc(sizeof(long long) * g_size * dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  } else {
    op_printf("Unknown type in file %s for map %s\n", file, name);
    exit(2);
  }

  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);

  free((char*)dset_props.type_str);

  op_map new_map = op_decl_map(from, to, dim, map, name);
  new_map->user_managed = 0;
  return new_map;
}

/*******************************************************************************
* Routine to read an op_dat from an hdf5 file
*******************************************************************************/

op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file,
                        char const *name) {
  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t attr;      // attribute identifier
  herr_t status;

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_dat_hdf5()\n",
              file);
    exit(2);
  }
  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);

  /* Save old error handler */
  H5E_auto_t old_func;
  void *old_client_data;
  H5error_off(&old_func, &old_client_data);

  /*open data set*/
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("op_dat with name : %s not found in file : %s \n", name, file);
    H5Fclose(file_id);
    return NULL;
  }

  op_hdf5_dataset_properties dset_props;
  status = get_dataset_properties(dset_id, &dset_props);
  if (status < 0) {
    op_printf("Could not get properties of dataset '%s' in file '%s'\n", name, file);
    exit(2);
  }

  int dat_dim = dset_props.dim;
  if (dat_dim != dim) {
    op_printf("dat.dim %d in file %s and dim %d do not match\n", dat_dim, file,
              dim);
    exit(2);
  }

  const char *typ = dset_props.type_str;
  if (!op_type_equivalence(typ, type)) {
    op_printf("dat.type %s in file %s and type %s do not match, performing automatic conversion\n", typ, file,
              type);
  }

  size_t type_size;

  // Restore previous error handler .. report hdf5 error stack automatically
  H5error_on(old_func, old_client_data);

  // Create the dataset with default properties
  dataspace = H5Dget_space(dset_id);

  // initialize data buffer and read in data
  char *data;
  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(double));
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    type_size = sizeof(double);

  } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
             strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(float));
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    type_size = sizeof(float);
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
             strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(int));
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    type_size = sizeof(int);
  } else if (strcmp(type, "long") == 0 || strcmp(type, "long:soa") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(long));
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    type_size = sizeof(long);
  } else if (strcmp(type, "long long") == 0 ||
             strcmp(type, "long long:soa") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(long long));
    H5Dread(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    type_size = sizeof(long long);
  } else {
    op_printf("unknown type for dat\n");
    exit(2);
  }

  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);

  free((char*)dset_props.type_str);

  op_dat new_dat = op_decl_dat_char(set, dim, type, type_size, data, name);
  new_dat->user_managed = 0;
  return new_dat;
}

/*******************************************************************************
* Routine to write all to a named hdf5 file
*******************************************************************************/

void op_dump_to_hdf5(char const *file_name) {
  op_printf("Writing to %s\n", file_name);

  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  op_timers(&cpu_t1, &wall_t1); // timer start for hdf5 file write

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier

  hsize_t dimsf[2]; // dataset dimensions

  file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  /*loop over all the op_sets and write them to file*/
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    // Create the dataspace for the dataset.
    hsize_t dimsf_set[] = {1};
    dataspace = H5Screate_simple(1, dimsf_set, NULL);

    // Create the dataset with default properties and close dataspace.
    dset_id = H5Dcreate(file_id, set->name, H5T_NATIVE_INT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    int size = set->size;

    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &size);
    H5Sclose(dataspace);
    H5Dclose(dset_id);
  }

  /*loop over all the op_maps and write them to file*/
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];

    // find total size of map
    int g_size = map->from->size;

    // Create the dataspace for the dataset.
    dimsf[0] = g_size;
    dimsf[1] = map->dim;
    dataspace = H5Screate_simple(2, dimsf, NULL);

    // create map path
    create_path(map->name, file_id);

    // Create the dataset with default properties and write data
    if (sizeof(map->map[0]) == sizeof(int)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
               map->map);
    } else if (sizeof(map->map[0]) == sizeof(long)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_LONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
               map->map);
    } else if (sizeof(map->map[0]) == sizeof(long long)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_LLONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
               map->map);
    } else {
      op_printf("Unknown type for map elements\n");
      exit(2);
    }

    H5Sclose(dataspace);
    H5Dclose(dset_id);

    /*attach attributes to map*/

    // open existing data set
    dset_id = H5Dopen(file_id, map->name, H5P_DEFAULT);
    // create the data space for the attribute
    hsize_t dims = 1;
    dataspace = H5Screate_simple(1, &dims, NULL);

    // Create an int attribute - size
    hid_t attribute = H5Acreate(dset_id, "size", H5T_NATIVE_INT, dataspace,
                                H5P_DEFAULT, H5P_DEFAULT);
    // Write the attribute data.
    H5Awrite(attribute, H5T_NATIVE_INT, &g_size);
    // Close the attribute.
    H5Aclose(attribute);

    // Create an int attribute - dimension
    attribute = H5Acreate(dset_id, "dim", H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT);
    // Write the attribute data.
    H5Awrite(attribute, H5T_NATIVE_INT, &map->dim);
    // Close the attribute.
    H5Aclose(attribute);
    H5Sclose(dataspace);

    // Create a string attribute - type
    dataspace = H5Screate(H5S_SCALAR);
    hid_t atype = H5Tcopy(H5T_C_S1);
    H5Tset_size(atype, 10);
    attribute =
        H5Acreate(dset_id, "type", atype, dataspace, H5P_DEFAULT, H5P_DEFAULT);

    if (sizeof(map->map[0]) == sizeof(int))
      H5Awrite(attribute, atype, "int");
    else if (sizeof(map->map[0]) == sizeof(long))
      H5Awrite(attribute, atype, "long");
    else if (sizeof(map->map[0]) == sizeof(long long))
      H5Awrite(attribute, atype, "long long");
    else {
      op_printf("Unknown type for map elements\n");
      exit(2);
    }

    H5Aclose(attribute);
    // Close the dataspace
    H5Sclose(dataspace);
    // Close to the dataset.
    H5Dclose(dset_id);
  }

  /*loop over all the op_dats and write them to file*/
  op_dat_entry *item;
  TAILQ_FOREACH(item, &OP_dat_list, entries) {
    op_dat dat = item->dat;
    if (dat->size == 0 || dat->data == NULL)
      continue;
    // find total size of dat
    int g_size = dat->set->size;

    // Create the dataspace for the dataset.
    dimsf[0] = g_size;
    dimsf[1] = dat->dim;
    dataspace = H5Screate_simple(2, dimsf, NULL);

    // create dateset path
    create_path(dat->name, file_id);

    // Create the dataset with default properties and write data
    if (strcmp(dat->type, "double") == 0 ||
        strcmp(dat->type, "double:soa") == 0 ||
        strcmp(dat->type, "double precision") == 0 ||
        strcmp(dat->type, "real(8)") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_DOUBLE, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
               dat->data);
    } else if (strcmp(dat->type, "float") == 0 ||
               strcmp(dat->type, "float:soa") == 0 ||
               strcmp(dat->type, "real(4)") == 0 ||
               strcmp(dat->type, "real") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_FLOAT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
               dat->data);
    } else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int:soa") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer") == 0 ||
               strcmp(dat->type, "integer(4)") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
               dat->data);
    } else if ((strcmp(dat->type, "long") == 0) ||
               (strcmp(dat->type, "long:soa") == 0)) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
               dat->data);
    } else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "long long:soa") == 0)) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LLONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
               dat->data);
    } else {
      op_printf("Unknown type for data elements %s\n", dat->type);
      exit(2);
    }

    H5Sclose(dataspace);
    H5Dclose(dset_id);

    /*attach attributes to dat*/

    // open existing data set
    dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);
    // create the data space for the attribute
    hsize_t dims = 1;
    dataspace = H5Screate_simple(1, &dims, NULL);

    // Create an int attribute - size
    hid_t attribute = H5Acreate(dset_id, "size", H5T_NATIVE_INT, dataspace,
                                H5P_DEFAULT, H5P_DEFAULT);
    // Write the attribute data.
    H5Awrite(attribute, H5T_NATIVE_INT, &dat->size);
    // Close the attribute.
    H5Aclose(attribute);

    // Create an int attribute - dimension
    attribute = H5Acreate(dset_id, "dim", H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT);
    // Write the attribute data.
    H5Awrite(attribute, H5T_NATIVE_INT, &dat->dim);
    H5Aclose(attribute);
    H5Sclose(dataspace);

    // Create an string attribute - type
    dataspace = H5Screate(H5S_SCALAR);
    hid_t atype = H5Tcopy(H5T_C_S1);
    int attlen = strlen(dat->type);
    H5Tset_size(atype, attlen);

    attribute =
        H5Acreate(dset_id, "type", atype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, atype, dat->type);
    H5Aclose(attribute);

    // Close the dataspace.
    H5Sclose(dataspace);
    H5Dclose(dset_id);
  }
  H5Fclose(file_id);
  op_timers(&cpu_t2, &wall_t2); // timer stop for hdf5 file write
}

/*******************************************************************************
* Routine to read in a constant from a named hdf5 file
*******************************************************************************/

void op_get_const_hdf5(char const *name, int dim, char const *type,
                       char *const_data, char const *file_name) {
  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t attr;      // attribute identifier

  if (file_exist(file_name) == 0) {
    op_printf("File %s does not exist .... aborting op_get_const_hdf5()\n",
              file_name);
    exit(2);
  }

  file_id = H5Fopen(file_name, H5F_ACC_RDONLY, H5P_DEFAULT);

  /// find dimension of this constant with available attributes
  int const_dim = 0;

  // open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("dataset '%s' not found in file '%s'\n", name, file_name);
    H5Fclose(file_id);
    const_data = NULL;
    return;
  }

  // get OID of the dim attribute
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &const_dim);
  H5Aclose(attr);
  H5Dclose(dset_id);
  if (const_dim != dim) {
    op_printf("dim of constant %d in file %s and requested dim %d do not match\n",
              const_dim, file_name, dim);
    exit(2);
  }

  // find type with available attributes
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);

  int attlen = H5Aget_storage_size(attr);
  H5Tset_size(atype, attlen + 1);

  // read attribute
  char typ[attlen + 1];
  H5Aread(attr, atype, typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  if (!op_type_equivalence(typ, type)) {
    op_printf(
        "type of constant %s in file %s and requested type %s do not match, performing automatic type conversion\n",
        typ, file_name, type);
    strcpy(typ,type);
  }

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  char *data;
  // initialize data buffer and read data
  if (strcmp(typ, "int") == 0 || strcmp(typ, "int(4)") == 0 ||
      strcmp(typ, "integer") == 0 || strcmp(typ, "integer(4)") == 0) {
    data = (char *)xmalloc(sizeof(int) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(int) * const_dim);
  } else if (strcmp(typ, "long") == 0) {
    data = (char *)xmalloc(sizeof(long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(long) * const_dim);
  } else if (strcmp(typ, "long long") == 0) {
    data = (char *)xmalloc(sizeof(long long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(long long) * const_dim);
  } else if (strcmp(typ, "float") == 0 || strcmp(typ, "real(4)") == 0 ||
             strcmp(typ, "real") == 0) {
    data = (char *)xmalloc(sizeof(float) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(float) * const_dim);
  } else if (strcmp(typ, "double") == 0 ||
             strcmp(typ, "double precision") == 0 ||
             strcmp(typ, "real(8)") == 0) {
    data = (char *)xmalloc(sizeof(double) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    memcpy((void *)const_data, (void *)data, sizeof(double) * const_dim);
  } else {
    op_printf("Unknown type in file %s for constant %s\n", file_name, name);
    exit(2);
  }

  free(data);

  H5Dclose(dset_id);
  H5Fclose(file_id);
}

/*******************************************************************************
* Routine to write a constant to a named hdf5 file
*******************************************************************************/

void op_write_const_hdf5(char const *name, int dim, char const *type,
                         char *const_data, char const *file_name) {
  // letting know that writing is happening ...
  op_printf("Writing '%s' to file '%s'\n", name, file_name);

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier

  if (file_exist(file_name) == 0) {
    if (OP_diags > 3) {
      op_printf("File %s does not exist .... creating file\n", file_name);
    }
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(file_id);
  }

  op_printf("Writing constant to %s\n", file_name);

  /* Open the existing file. */
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);

  // Create the dataspace for the dataset.
  hsize_t dims_of_const = {(hsize_t)dim};
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create the dataset with default properties
  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
             strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
             strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long") == 0) || (strcmp(type, "long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long long") == 0) ||
             (strcmp(type, "long long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
             const_data);
    H5Dclose(dset_id);
  } else {
    op_printf("Unknown type for write_const\n");
    exit(2);
  }

  H5Sclose(dataspace);

  /*attach attributes to constant*/

  // open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  // create the data space for the attribute
  dims_of_const = 1;
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create an int attribute - dimension
  hid_t attribute = H5Acreate(dset_id, "dim", H5T_NATIVE_INT, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data.
  H5Awrite(attribute, H5T_NATIVE_INT, &dim);
  // Close the attribute.
  H5Aclose(attribute);
  H5Sclose(dataspace);

  // Create a string attribute - type
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);

  int attlen = strlen(type);
  H5Tset_size(atype, attlen);
  attribute =
      H5Acreate(dset_id, "type", atype, dataspace, H5P_DEFAULT, H5P_DEFAULT);

  if (strcmp(type, "double") == 0 || strcmp(type, "double precision") == 0 ||
      strcmp(type, "real(8)") == 0)
    H5Awrite(attribute, atype, "double");
  else if (strcmp(type, "int") == 0 || strcmp(type, "int(4)") == 0 ||
           strcmp(type, "integer") == 0 || strcmp(type, "integer(4)") == 0)
    H5Awrite(attribute, atype, "int");
  else if (strcmp(type, "long") == 0)
    H5Awrite(attribute, atype, "long");
  else if (strcmp(type, "long long") == 0)
    H5Awrite(attribute, atype, "long long");
  else if (strcmp(type, "float") == 0 || strcmp(type, "real(4)") == 0 ||
           strcmp(type, "real") == 0)
    H5Awrite(attribute, atype, "float");
  else {
    op_printf("Unknown type %s for constant %s: cannot write constant to file\n",
              type, name);
    exit(2);
  }

  H5Aclose(attribute);
  H5Sclose(dataspace);
  H5Dclose(dset_id);

  H5Fclose(file_id);
}

/*******************************************************************************
* Routine to write an op_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set given at path does not exists in file creates data set
*******************************************************************************/

void op_fetch_data_hdf5(op_dat dat, char const *file_name,
                        char const *path_name) {
  // letting know that writing is happening ...
  op_printf("Writing '%s' to file '%s'\n", path_name, file_name);

  // fetch data based on the backend
  op_fetch_data_char(dat, dat->data);

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t attr;      // attribute identifier

  hsize_t dimsf[2]; // dataset dimensions

  H5E_auto_t old_func;
  void *old_client_data;

  if (file_exist(file_name) == 0) {
    if (OP_diags > 3) {
      op_printf("File %s does not exist .... creating file\n", file_name);
    }
    file_id = H5Fcreate(file_name, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
  } else {
    if (OP_diags > 3) {
      op_printf("File %s exists .... checking for dataset and path %s in file\n",
                file_name, path_name);
    }
    file_id = H5Fopen(file_name, H5F_ACC_RDWR, H5P_DEFAULT);

    H5error_off(&old_func, &old_client_data);
    herr_t status = H5Gget_objinfo(file_id, path_name, 0, NULL);
    H5error_on(old_func, old_client_data);

    if (status == 0) {
      if (OP_diags > 3) {
        op_printf("dataset '%s' exists in the file ... updating data\n", path_name);
      }
      dset_id = H5Dopen(file_id, path_name, H5P_DEFAULT);

      // find dat->size with available attributes
      size_t dat_size = 0;
      attr = H5Aopen(dset_id, "size", H5P_DEFAULT);

      if (attr > 0) {
        H5Aread(attr, H5T_NATIVE_INT, &dat_size);
        H5Aclose(attr);
        if (dat_size != dat->size) {
          op_printf(
              "dat.size %zu in file %s and dim %d do not match ... aborting\n",
              dat_size, file_name, dat->dim);
          exit(2);
        }
      } else {
        op_printf("data set %s on file %s does not have attribute 'size'",
                  path_name, file_name);
        op_printf(" -- cannot check size ... aborting\n");
        exit(2);
      }

      // find dim with available attributes
      int dat_dim = 0;
      attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);

      if (attr > 0) {
        H5Aread(attr, H5T_NATIVE_INT, &dat_dim);
        H5Aclose(attr);
        if (dat_dim != dat->dim) {
          op_printf("dat.dim %d in file %s and dim %d do not match ... aborting\n",
                    dat_dim, file_name, dat->dim);
          exit(2);
        }
      } else {
        op_printf("data set %s on file %s does not have attribute 'dim'",
                  path_name, file_name);
        op_printf(" -- cannot check dim ... aborting\n");
        exit(2);
      }

      // find type with available attributes
      dataspace = H5Screate(H5S_SCALAR);
      hid_t atype = H5Tcopy(H5T_C_S1);
      attr = H5Aopen(dset_id, "type", H5P_DEFAULT);

      if (attr > 0) {
        int attlen = H5Aget_storage_size(attr);
        H5Tset_size(atype, attlen + 1);
        char typ[attlen + 1];
        H5Aread(attr, atype, typ);
        H5Aclose(attr);
        H5Sclose(dataspace);
        char typ_soa[50];
        sprintf(typ_soa, "%s:soa", typ);
        if (!op_type_equivalence(typ, dat->type)) {
          op_printf("dat.type %s in file %s and type %s do not match\n", typ,
                    file_name, dat->type);
          exit(2);
        }
      } else {
        op_printf("data set %s on file %s does not have attribute 'type'",
                  path_name, file_name);
        op_printf(" -- cannot check type ... aborting\n");
        exit(2);
      }

      //
      // all good .. we can update existing dat now
      //

      // Create the dataspace for the dataset.
      dimsf[0] = dat->set->size;
      dimsf[1] = dat->dim;
      dataspace = H5Screate_simple(2, dimsf, NULL);

      if (strcmp(dat->type, "double") == 0 ||
          strcmp(dat->type, "double:soa") == 0 ||
          strcmp(dat->type, "double precision") == 0 ||
          strcmp(dat->type, "real(8)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
                 dat->data);
      else if (strcmp(dat->type, "float") == 0 ||
               strcmp(dat->type, "float:soa") == 0 ||
               strcmp(dat->type, "real(4)") == 0 ||
               strcmp(dat->type, "real") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
                 dat->data);
      else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int:soa") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer") == 0 ||
               strcmp(dat->type, "integer(4)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
                 dat->data);
      else if ((strcmp(dat->type, "long") == 0) ||
               (strcmp(dat->type, "long:soa") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
                 dat->data);
      else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "long long:soa") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
                 dat->data);
      else {
        op_printf("Unknown type for data elements\n");
        exit(2);
      }

      H5Dclose(dset_id);
      H5Sclose(dataspace);
      H5Fclose(file_id);
      return;
    } else {
      if (OP_diags > 3) {
        op_printf("op_dat %s does not exists in the file ... creating data set\n",
                  path_name);
      }
    }
  }

  //
  // new data set ...
  //

  // Create the dataspace for the dataset.
  dimsf[0] = dat->set->size;
  dimsf[1] = dat->dim;
  dataspace = H5Screate_simple(2, dimsf, NULL);

  // create dataset path
  create_path(path_name, file_id);

 // Create the dataset with default properties and write data
  if ((strcmp(dat->type, "double") == 0) ||
      (strcmp(dat->type, "double:soa") == 0) ||
      (strcmp(dat->type, "double precision") == 0) ||      
      (strcmp(dat->type, "real(8)") == 0)) {
    dset_id = H5Dcreate(file_id, path_name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, H5P_DEFAULT,
             dat->data);
  } else if ((strcmp(dat->type, "float") == 0) ||
             (strcmp(dat->type, "float:soa") == 0) ||
             (strcmp(dat->type, "real(4)") == 0) ||
             (strcmp(dat->type, "real") == 0)) {
    dset_id = H5Dcreate(file_id, path_name, H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, H5P_DEFAULT,
             dat->data);
  } else if ((strcmp(dat->type, "int") == 0) ||
             (strcmp(dat->type, "int:soa") == 0) ||
             (strcmp(dat->type, "int(4)") == 0) ||
             (strcmp(dat->type, "integer") == 0) ||
             (strcmp(dat->type, "integer(4)") == 0)) {
    dset_id = H5Dcreate(file_id, path_name, H5T_NATIVE_INT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, H5P_DEFAULT,
             dat->data);
  } else if ((strcmp(dat->type, "long") == 0) ||
             (strcmp(dat->type, "long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, path_name, H5T_NATIVE_LONG, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, H5P_DEFAULT,
             dat->data);
  } else if ((strcmp(dat->type, "long long") == 0) ||
             (strcmp(dat->type, "long long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, path_name, H5T_NATIVE_LLONG, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, H5P_DEFAULT,
             dat->data);
  } else {
    op_printf("Unknown type for data elements\n");
    exit(2);
  }

  H5Sclose(dataspace);
  H5Dclose(dset_id);

  /*attach attributes to dat*/

  // open existing data set
  dset_id = H5Dopen(file_id, path_name, H5P_DEFAULT);

  // create the data space for the attribute
  hsize_t dims = 1;
  dataspace = H5Screate_simple(1, &dims, NULL);

  // Create an int attribute - size
  hid_t attribute = H5Acreate(dset_id, "size", H5T_NATIVE_INT, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attribute, H5T_NATIVE_INT, &dat->size);
  H5Aclose(attribute);

  // Create an int attribute - dimension
  attribute = H5Acreate(dset_id, "dim", H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT);
  H5Awrite(attribute, H5T_NATIVE_INT, &dat->dim);
  H5Aclose(attribute);
  H5Sclose(dataspace);

  // Create an string attribute - type
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  int attlen = strlen(dat->type);
  H5Tset_size(atype, attlen);

  attribute =
      H5Acreate(dset_id, "type", atype, dataspace, H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attribute, atype, dat->type);
  H5Aclose(attribute);

  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);
}

/*******************************************************************************
* Routine to write an op_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set does not exists in file creates data set
*******************************************************************************/

void op_fetch_data_hdf5_file(op_dat dat, char const *file_name) {
  op_fetch_data_hdf5(dat, file_name, dat->name);
}

void op_fetch_data_hdf5_file_ptr(char *data, const char *file_name) {
  op_dat_entry *item;
  op_dat_entry *tmp_item;
  op_dat item_dat = NULL;
  for (item = TAILQ_FIRST(&OP_dat_list); item != NULL; item = tmp_item) {
    tmp_item = TAILQ_NEXT(item, entries);
    // printf("Available op_dat %s with pointer %p\n", item->dat->name,
    // item->dat->data);
    if (item->orig_ptr == data) {
      // printf("%s(%p), ", item->dat->name, item->dat->data);
      item_dat = item->dat;
      break;
    }
  }
  // printf("\n");
  if (item_dat == NULL) {
    printf("ERROR in op_partition: op_dat not found for dat with %p pointer\n",
           data);
  }

  op_fetch_data_hdf5(item_dat, file_name, item_dat->name);
}

/*******************************************************************************
* Routine to write an op_dat to a named hdf5 file, where dataset_name is a
* freely chosen path inside the h5-file of the dataset.
* If file does not exist, creates it
* If the data set does not exists in file creates data set
*******************************************************************************/

void op_fetch_data_hdf5_file_path(op_dat dat, char const *file_name,
                                  char const *path_name) {
  op_fetch_data_hdf5(dat, file_name, path_name);
}
