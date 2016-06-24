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
 * op_mpi_core.c
 *
 * Implements the HDF5 based parallel I/O routines for the OP2 Distributed
 * memory
 * backe end
 *
 * written by: Gihan R. Mudalige, (Started 10-10-2011)
 */

// mpi header
#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>
#include <op_util.h>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2
// hdf5 header
#include <hdf5.h>

#include <H5FDmpio.h>
#include <op_hdf5.h>
#include <op_lib_mpi.h>

//
// MPI Communicator for parallel I/O
//

MPI_Comm OP_MPI_HDF5_WORLD;

int compute_local_size_weight(int global_size, int mpi_comm_size,
                              int mpi_rank) {
  int *hybrid_flags = (int *)xmalloc(mpi_comm_size * sizeof(int));
  MPI_Allgather(&OP_hybrid_gpu, 1, MPI_INT, hybrid_flags, 1, MPI_INT,
                OP_MPI_HDF5_WORLD);
  double total = 0;
  for (int i = 0; i < mpi_comm_size; i++)
    total += hybrid_flags[i] == 1 ? OP_hybrid_balance : 1.0;
  double cumulative = 0;
  for (int i = 0; i < mpi_rank; i++)
    cumulative += hybrid_flags[i] == 1 ? OP_hybrid_balance : 1.0;

  int local_start = ((double)global_size) * (cumulative / total);
  int local_end =
      ((double)global_size) *
      ((cumulative + (OP_hybrid_gpu ? OP_hybrid_balance : 1.0)) / total);
  if (mpi_rank + 1 == mpi_comm_size)
    local_end = global_size; // make sure we don't have rounding problems
  int local_size = local_end - local_start;

  return local_size;
}

/*******************************************************************************
* Routine to read an op_set from an hdf5 file
*******************************************************************************/

op_set op_decl_set_hdf5(char const *file, char const *name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;  // file identifier
  hid_t plist_id; // property list identifier
  hid_t dset_id;  // dataset identifier

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_set_hdf5()\n",
              file);
    MPI_Abort(OP_MPI_HDF5_WORLD, 2);
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  file_id = H5Fopen(file, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

  // Create property list for collective dataset read.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  int g_size = 0;
  // read data
  H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plist_id, &g_size);

  H5Pclose(plist_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);

  // calculate local size of set for this mpi process
  int l_size = compute_local_size_weight(g_size, comm_size, my_rank);
  MPI_Comm_free(&OP_MPI_HDF5_WORLD);

  return op_decl_set(l_size, name);
}

/*******************************************************************************
* Routine to read an op_map from an hdf5 file
*******************************************************************************/

op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file,
                        char const *name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t plist_id;  // property list identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t memspace;  // memory space identifier

  hsize_t count[2]; // hyperslab selection parameters
  hsize_t offset[2];

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_map_hdf5()\n",
              file);
    return NULL;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);
  file_id = H5Fopen(file, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  /* Save old error handler */
  H5E_auto_t old_func;
  void *old_client_data;
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL); // turn off HDF5's auto error reporting

  /*open data set*/
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("op_map with name : %s not found in file : %s \n", name, file);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return NULL;
  }

  /*find total size of this map by reading attributes*/
  int g_size;
  hid_t attr = H5Aopen(dset_id, "size", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_map with name : %s does not have attribute : %s \n", name,
              "size");
    return NULL;
  }
  H5Aread(attr, H5T_NATIVE_INT, &g_size);
  H5Aclose(attr);
  // calculate local size of set for this mpi process
  int l_size = compute_local_size_weight(g_size, comm_size, my_rank);
  // check if size is accurate
  if (from->size != l_size) {
    op_printf(
        "map from set size %d in file %s and size %d do not match on rank %d\n",
        l_size, file, from->size, my_rank);
    return NULL;
  }

  /*find dim with available attributes*/
  int map_dim = 0;
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_map with name : %s does not have attribute : %s \n", name,
              "dim");
    return NULL;
  }
  H5Aread(attr, H5T_NATIVE_INT, &map_dim);
  H5Aclose(attr);
  if (map_dim != dim) {
    op_printf("map.dim %d in file %s and dim %d do not match\n", map_dim, file,
              dim);
    return NULL;
  }

  /*find type with available attributes*/
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, 10);
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_map with name : %s does not have attribute : %s \n", name,
              "type");
    return NULL;
  }
  char typ[10];
  H5Aread(attr, atype, typ);
  H5Aclose(attr);
  H5Sclose(dataspace);

  // Restore previous error handler .. report hdf5 error stack automatically
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

  /*read in map in hyperslabs*/

  // Each process defines dataset in memory and reads from a hyperslab in the
  // file.
  int disp = 0;
  int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
  MPI_Allgather(&l_size, 1, MPI_INT, sizes, 1, MPI_INT, OP_MPI_HDF5_WORLD);
  for (int i = 0; i < my_rank; i++)
    disp = disp + sizes[i];
  op_free(sizes);

  count[0] = l_size;
  count[1] = dim;
  offset[0] = disp;
  offset[1] = 0;
  memspace = H5Screate_simple(2, count, NULL);

  // Select hyperslab in the file.
  dataspace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // initialize data buffer and read data
  int *map = 0;
  if (strcmp(typ, "int") == 0 || strcmp(typ, "integer(4)") == 0) {
    map = (int *)xmalloc(sizeof(int) * l_size * dim);
    H5Dread(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id, map);
  } else if (strcmp(typ, "long") == 0) {
    map = (int *)xmalloc(sizeof(long) * l_size * dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id, map);
  } else if (strcmp(typ, "long long") == 0) {
    map = (int *)xmalloc(sizeof(long) * l_size * dim);
    H5Dread(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id, map);
  } else {
    op_printf("unknown type\n");
    return NULL;
  }

  H5Pclose(plist_id);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OP_MPI_HDF5_WORLD);

  op_map new_map = op_decl_map_core(from, to, dim, map, name);
  new_map->user_managed = 0;
  return new_map;
}

/*******************************************************************************
* Routine to read an op_dat from an hdf5 file
*******************************************************************************/

op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file,
                        char const *name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t plist_id;  // property list identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t memspace;  // memory space identifier

  hsize_t count[2]; // hyperslab selection parameters
  hsize_t offset[2];
  hid_t attr; // attribute identifier

  if (file_exist(file) == 0) {
    op_printf("File %s does not exist .... aborting op_decl_dat_hdf5()\n",
              file);
    return NULL;
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  file_id = H5Fopen(file, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  /* Save old error handler */
  H5E_auto_t old_func;
  void *old_client_data;
  H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL); // turn off HDF5's auto error reporting

  /*open data set*/
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  if (dset_id < 0) {
    op_printf("op_dat with name : %s not found in file : %s \n", name, file);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return NULL;
  }

  /*find element size of this dat with available attributes*/
  size_t dat_size = 0;
  attr = H5Aopen(dset_id, "size", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_dat with name : %s does not have attribute : %s \n", name,
              "size");
    return NULL;
  }
  H5Aread(attr, H5T_NATIVE_INT, &dat_size);
  H5Aclose(attr);

  /*find dim with available attributes*/
  int dat_dim = 0;
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_dat with name : %s does not have attribute : %s \n", name,
              "dim");
    return NULL;
  }
  H5Aread(attr, H5T_NATIVE_INT, &dat_dim);
  H5Aclose(attr);
  if (dat_dim != dim) {
    op_printf("dat.dim %d in file %s and dim %d do not match\n", dat_dim, file,
              dim);
    return NULL;
  }

  /*find type with available attributes*/
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);
  if (attr < 0) {
    op_printf("op_dat with name : %s does not have attribute : %s \n", name,
              "type");
    return NULL;
  }
  // get length of attribute
  int attlen = H5Aget_storage_size(attr);
  H5Tset_size(atype, attlen + 1);

  char typ[10];
  H5Aread(attr, atype, typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  if (!op_type_equivalence(typ, type)) {
    op_printf("dat.type %s in file %s and type %s do not match\n", typ, file,
              type);
    return NULL;
  }

  // Restore previous error handler .. report hdf5 error stack automatically
  H5Eset_auto(H5E_DEFAULT, old_func, old_client_data);

  /*read in dat in hyperslabs*/

  // Create the dataset with default properties and close dataspace.
  // Each process defines dataset in memory and reads from a hyperslab in the
  // file.
  int disp = 0;
  int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
  MPI_Allgather(&(set->size), 1, MPI_INT, sizes, 1, MPI_INT, OP_MPI_HDF5_WORLD);
  for (int i = 0; i < my_rank; i++)
    disp = disp + sizes[i];
  op_free(sizes);

  count[0] = set->size;
  count[1] = dim;
  offset[0] = disp;
  offset[1] = 0;
  memspace = H5Screate_simple(2, count, NULL);

  // Select hyperslab in the file.
  dataspace = H5Dget_space(dset_id);
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // initialize data buffer and read in data
  char *data = 0;
  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(double));
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist_id, data);

    if (dat_size != dim * sizeof(double)) {
      op_printf("dat.size %lu in file %s and %d*sizeof(double) do not match\n",
                dat_size, file, dim);
      return NULL;
    } else
      dat_size = sizeof(double);
  } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
             strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(float));
    H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id, data);

    if (dat_size != dim * sizeof(float)) {
      op_printf("dat.size %lu in file %s and %d*sizeof(float) do not match\n",
                dat_size, file, dim);
      return NULL;
    } else
      dat_size = sizeof(float);

  } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
             strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    data = (char *)xmalloc(set->size * dim * sizeof(int));
    H5Dread(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id, data);

    if (dat_size != dim * sizeof(int)) {
      op_printf("dat.size %lu in file %s and %d*sizeof(int) do not match\n",
                dat_size, file, dim);
      return NULL;
    } else
      dat_size = sizeof(int);
  } else {
    op_printf("unknown type\n");
    return NULL;
  }

  H5Pclose(plist_id);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OP_MPI_HDF5_WORLD);

  op_dat new_dat = op_decl_dat_core(set, dim, type, dat_size, data, name);
  new_dat->user_managed = 0;
  return new_dat;
}

/*******************************************************************************
* Routine to read in a constant from a named hdf5 file
*******************************************************************************/
void op_get_const_hdf5(char const *name, int dim, char const *type,
                       char *const_data, char const *file_name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t plist_id;  // property list identifier
  hid_t dset_id;   // dataset identifier
  hid_t dataspace; // data space identifier
  hid_t attr;      // attribute identifier

  if (file_exist(file_name) == 0) {
    op_printf("File %s does not exist .... aborting op_get_const_hdf5()\n",
              file_name);
    MPI_Abort(OP_MPI_HDF5_WORLD, 2);
  }

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  file_id = H5Fopen(file_name, H5F_ACC_RDONLY, plist_id);
  H5Pclose(plist_id);

  // find dimension of this constant with available attributes
  int const_dim = 0;
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

  // Create property list for collective dataset read.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // get OID of the dim attribute
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  H5Aread(attr, H5T_NATIVE_INT, &const_dim);
  H5Aclose(attr);
  H5Dclose(dset_id);
  if (const_dim != dim) {
    printf("dim of constant %d in file %s and requested dim %d do not match\n",
           const_dim, file_name, dim);
    MPI_Abort(OP_MPI_HDF5_WORLD, 2);
  }

  // find type with available attributes
  dataspace = H5Screate(H5S_SCALAR);
  hid_t atype = H5Tcopy(H5T_C_S1);
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);

  int attlen = H5Aget_storage_size(attr);
  H5Tset_size(atype, attlen + 1);

  char typ[attlen + 1];
  H5Aread(attr, atype, typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  if (!op_type_equivalence(typ, type)) {
    printf(
        "type of constant %s in file %s and requested type %s do not match\n",
        typ, file_name, type);
    exit(2);
  }

  // Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  char *data;
  // initialize data buffer and read data
  if (strcmp(typ, "int") == 0 || strcmp(typ, "int(4)") == 0 ||
      strcmp(typ, "integer") == 0 || strcmp(typ, "integer(4)") == 0) {
    data = (char *)xmalloc(sizeof(int) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(int) * const_dim);
  } else if (strcmp(typ, "long") == 0) {
    data = (char *)xmalloc(sizeof(long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(long) * const_dim);
  } else if (strcmp(typ, "long long") == 0) {
    data = (char *)xmalloc(sizeof(long long) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(long long) * const_dim);
  } else if (strcmp(typ, "float") == 0 || strcmp(typ, "real(4)") == 0 ||
             strcmp(typ, "real") == 0) {
    data = (char *)xmalloc(sizeof(float) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(float) * const_dim);
  } else if (strcmp(typ, "double") == 0 ||
             strcmp(typ, "double precision") == 0 ||
             strcmp(typ, "real(8)") == 0) {
    data = (char *)xmalloc(sizeof(double) * const_dim);
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, plist_id, data);
    memcpy((void *)const_data, (void *)data, sizeof(double) * const_dim);
  } else {
    printf("Unknown type in file %s for constant %s\n", file_name, name);
    exit(2);
  }

  op_free(data);

  H5Pclose(plist_id);
  H5Dclose(dset_id);
  H5Fclose(file_id);
  MPI_Comm_free(&OP_MPI_HDF5_WORLD);
}

/*******************************************************************************
* Routine to write all to a named hdf5 file
*******************************************************************************/
void op_dump_to_hdf5(char const *file_name) {
  op_printf("Writing to %s\n", file_name);

  // declare timers
  double cpu_t1, cpu_t2, wall_t1, wall_t2;
  double time;
  double max_time;
  op_timers(&cpu_t1, &wall_t1); // timer start for hdf5 file write
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;     // file identifier
  hid_t plist_id;    // property list identifier
  hid_t dset_id = 0; // dataset identifier
  hid_t dataspace;   // data space identifier
  hid_t memspace;    // memory space identifier

  hsize_t dimsf[2]; // dataset dimensions
  hsize_t count[2]; // hyperslab selection parameters
  hsize_t offset[2];

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  // Create a new file collectively and release property list identifier.
  file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  /*loop over all the op_sets and write them to file*/
  for (int s = 0; s < OP_set_index; s++) {
    op_set set = OP_set_list[s];

    // Create the dataspace for the dataset.
    hsize_t dimsf_set[] = {1};
    dataspace = H5Screate_simple(1, dimsf_set, NULL);

    // Create the dataset with default properties and close dataspace.
    dset_id = H5Dcreate(file_id, set->name, H5T_NATIVE_INT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    int size = 0;
    int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
    MPI_Allgather(&set->size, 1, MPI_INT, sizes, 1, MPI_INT, OP_MPI_HDF5_WORLD);
    for (int i = 0; i < comm_size; i++)
      size = size + sizes[i];

    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, plist_id, &size);
    H5Sclose(dataspace);
    H5Pclose(plist_id);
    H5Dclose(dset_id);
  }
  /*loop over all the op_maps and write them to file*/
  for (int m = 0; m < OP_map_index; m++) {
    op_map map = OP_map_list[m];

    if (map->dim == 0)
      continue;
    // find total size of map
    int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
    int g_size = 0;
    MPI_Allgather(&map->from->size, 1, MPI_INT, sizes, 1, MPI_INT,
                  OP_MPI_HDF5_WORLD);
    for (int i = 0; i < comm_size; i++)
      g_size = g_size + sizes[i];
    if (g_size == 0)
      continue;
    // Create the dataspace for the dataset.
    dimsf[0] = g_size;
    dimsf[1] = map->dim;
    dataspace = H5Screate_simple(2, dimsf, NULL);

    // Each process defines dataset in memory and writes it to a hyperslab
    // in the file.
    int disp = 0;
    for (int i = 0; i < my_rank; i++)
      disp = disp + sizes[i];
    count[0] = map->from->size;
    count[1] = dimsf[1];
    offset[0] = disp;
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);

    // Select hyperslab in the file.
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // Create the dataset with default properties and close dataspace.
    if (sizeof(map->map[0]) == sizeof(int)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id,
               map->map);
    } else if (sizeof(map->map[0]) == sizeof(long)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_LONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id,
               map->map);
    } else if (sizeof(map->map[0]) == sizeof(long long)) {
      dset_id = H5Dcreate(file_id, map->name, H5T_NATIVE_LLONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id,
               map->map);
    }

    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    op_free(sizes);

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
    if (sizeof(map->map[0]) == sizeof(long))
      H5Awrite(attribute, atype, "long");
    if (sizeof(map->map[0]) == sizeof(long long))
      H5Awrite(attribute, atype, "long long");

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
    if (dat->size == 0)
      continue;
    // find total size of dat
    int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
    int g_size = 0;
    MPI_Allgather(&dat->set->size, 1, MPI_INT, sizes, 1, MPI_INT,
                  OP_MPI_HDF5_WORLD);
    for (int i = 0; i < comm_size; i++)
      g_size = g_size + sizes[i];
    if (g_size == 0)
      continue;
    // Create the dataspace for the dataset.
    dimsf[0] = g_size;
    dimsf[1] = dat->dim;
    dataspace = H5Screate_simple(2, dimsf, NULL);

    // Each process defines dataset in memory and writes it to a hyperslab
    // in the file.
    int disp = 0;
    for (int i = 0; i < my_rank; i++)
      disp = disp + sizes[i];
    count[0] = dat->set->size;
    count[1] = dimsf[1];
    offset[0] = disp;
    offset[1] = 0;
    memspace = H5Screate_simple(2, count, NULL);

    // Select hyperslab in the file.
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Create property list for collective dataset write.
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    // Create the dataset with default properties and close dataspace.
    if (strcmp(dat->type, "double") == 0 ||
        strcmp(dat->type, "double:soa") == 0 ||
        strcmp(dat->type, "double precision") == 0 ||
        strcmp(dat->type, "real(8)") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_DOUBLE, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist_id,
               dat->data);
    } else if (strcmp(dat->type, "float") == 0 ||
               strcmp(dat->type, "float:soa") == 0 ||
               strcmp(dat->type, "real(4)") == 0 ||
               strcmp(dat->type, "real") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_FLOAT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id,
               dat->data);
    } else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int:soa") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer") == 0 ||
               strcmp(dat->type, "integer(4)") == 0) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_INT, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id,
               dat->data);
    } else if ((strcmp(dat->type, "long") == 0) ||
               (strcmp(dat->type, "long:soa") == 0)) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id,
               dat->data);
    } else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "long long:soa") == 0)) {
      dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LLONG, dataspace,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id,
               dat->data);
    } else {
      printf("Unknown type - in op_dump_to_hdf5() writing op_dats\n");
      MPI_Abort(OP_MPI_HDF5_WORLD, 2);
    }

    H5Dclose(dset_id);
    H5Pclose(plist_id);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    op_free(sizes);

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

    // Create a string attribute - type
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
  }
  H5Fclose(file_id);

  op_timers(&cpu_t2, &wall_t2); // timer stop for hdf5 file write
  time = wall_t2 - wall_t1;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MPI_ROOT,
             OP_MPI_HDF5_WORLD);

  if (my_rank == MPI_ROOT)
    printf("Max hdf5 file write time = %lf\n\n", max_time);

  MPI_Comm_free(&OP_MPI_HDF5_WORLD);
}

/*******************************************************************************
* Routine to write a constant to a named hdf5 file
*******************************************************************************/
void op_write_const_hdf5(char const *name, int dim, char const *type,
                         char *const_data, char const *file_name) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;   // file identifier
  hid_t dset_id;   // dataset identifier
  hid_t plist_id;  // property list identifier
  hid_t dataspace; // data space identifier

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    op_printf("File %s does not exist .... creating file\n", file_name);
    file_id = H5Fcreate(file_name, H5F_ACC_EXCL, H5P_DEFAULT, plist_id);
    H5Fclose(file_id);
  }

  op_printf("Writing constant to %s\n", file_name);

  /* Open the existing file. */
  file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);

  // Create the dataspace for the dataset.
  hsize_t dims_of_const = {dim};
  dataspace = H5Screate_simple(1, &dims_of_const, NULL);

  // Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // Create the dataset with default properties
  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
             strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
             strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
             strcmp(type, "integer(4)") == 0) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, dataspace, plist_id, const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long") == 0) || (strcmp(type, "long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LONG, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else if ((strcmp(type, "long long") == 0) ||
             (strcmp(type, "long long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, name, H5T_NATIVE_LLONG, dataspace, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
    // write data
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, H5S_ALL, dataspace, plist_id,
             const_data);
    H5Dclose(dset_id);
  } else
    printf("Unknown type in op_write_const_hdf5()\n");

  H5Pclose(plist_id);
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

  if (strcmp(type, "double") == 0 || strcmp(type, "double:soa") == 0 ||
      strcmp(type, "double precision") == 0 || strcmp(type, "real(8)") == 0)
    H5Awrite(attribute, atype, "double");
  else if (strcmp(type, "int") == 0 || strcmp(type, "int:soa") == 0 ||
           strcmp(type, "int(4)") == 0 || strcmp(type, "integer") == 0 ||
           strcmp(type, "integer(4)") == 0)
    H5Awrite(attribute, atype, "int");
  else if (strcmp(type, "long") == 0)
    H5Awrite(attribute, atype, "long");
  else if (strcmp(type, "long long") == 0)
    H5Awrite(attribute, atype, "long long");
  else if (strcmp(type, "float") == 0 || strcmp(type, "float:soa") == 0 ||
           strcmp(type, "real(4)") == 0 || strcmp(type, "real") == 0)
    H5Awrite(attribute, atype, "float");
  else {
    printf("Unknown type %s for constant %s: cannot write constant to file\n",
           type, name);
    exit(2);
  }

  H5Aclose(attribute);
  H5Sclose(dataspace);
  H5Dclose(dset_id);

  H5Fclose(file_id);
  MPI_Comm_free(&OP_MPI_HDF5_WORLD);
}

/*******************************************************************************
* Routine to write an op_dat to a named hdf5 file,
* if file does not exist, creates it
* if the data set does not exists in file creates data set
*******************************************************************************/

void op_fetch_data_hdf5_file(op_dat data, char const *file_name) {
  // fetch data based on the backend
  op_dat dat = op_fetch_data_file_char(data);

  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_HDF5_WORLD);
  MPI_Comm_rank(OP_MPI_HDF5_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_HDF5_WORLD, &comm_size);

  // MPI variables
  MPI_Info info = MPI_INFO_NULL;

  // HDF5 APIs definitions
  hid_t file_id;     // file identifier
  hid_t dset_id = 0; // dataset identifier
  hid_t dataspace;   // data space identifier
  hid_t plist_id;    // property list identifier
  hid_t memspace;    // memory space identifier
  hid_t attr;        // attribute identifier

  hsize_t dimsf[2]; // dataset dimensions
  hsize_t count[2]; // hyperslab selection parameters
  hsize_t offset[2];

  // Set up file access property list with parallel I/O access
  plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, OP_MPI_HDF5_WORLD, info);

  if (file_exist(file_name) == 0) {
    MPI_Barrier(OP_MPI_WORLD);
    op_printf("File %s does not exist .... creating file\n", file_name);
    MPI_Barrier(OP_MPI_HDF5_WORLD);
    if (op_is_root()) {
      FILE *fp;
      fp = fopen(file_name, "w");
      fclose(fp);
    }
    MPI_Barrier(OP_MPI_HDF5_WORLD);
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  } else {
    op_printf("File %s exists .... checking for dataset %s in file\n",
              file_name, dat->name);
    file_id = H5Fopen(file_name, H5F_ACC_RDWR, plist_id);
    if (H5Lexists(file_id, dat->name, H5P_DEFAULT) != 0) {
      op_printf("op_dat %s exists in the file ... updating data\n", dat->name);

      dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);

      // find element size of this dat with available attributes
      size_t dat_size = 0;
      attr = H5Aopen(dset_id, "size", H5P_DEFAULT);

      if (attr > 0) {
        H5Aread(attr, H5T_NATIVE_INT, &dat_size);
        H5Aclose(attr);
        if (dat_size != dat->size) {
          printf(
              "dat.size %zu in file %s and dim %d do not match ... aborting\n",
              dat_size, file_name, dat->dim);
          MPI_Abort(OP_MPI_HDF5_WORLD, 2);
        }
      } else {
        printf("data set %s on file %s does not have attribute 'size'",
               dat->name, file_name);
        printf(" -- cannot check size ... aborting\n");
        MPI_Abort(OP_MPI_HDF5_WORLD, 2);
      }

      // find dim with available attributes
      int dat_dim = 0;
      attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);

      if (attr > 0) {
        H5Aread(attr, H5T_NATIVE_INT, &dat_dim);
        H5Aclose(attr);
        if (dat_dim != dat->dim) {
          printf("dat.dim %d in file %s and dim %d do not match ... aborting\n",
                 dat_dim, file_name, dat->dim);
          exit(2);
        }
      } else {
        printf("data set %s on file %s does not have attribute 'dim'",
               dat->name, file_name);
        printf(" -- cannot check dim ... aborting\n");
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
          printf("dat.type %s in file %s and type %s do not match\n", typ,
                 file_name, dat->type);
          exit(2);
        }
      } else {
        printf("data set %s on file %s does not have attribute 'type'",
               dat->name, file_name);
        printf(" -- cannot check type ... aborting\n");
        exit(2);
      }

      //
      // all good .. we can update existing dat now
      //

      // find total size of dat
      int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
      int g_size = 0;
      MPI_Allgather(&dat->set->size, 1, MPI_INT, sizes, 1, MPI_INT,
                    OP_MPI_HDF5_WORLD);
      for (int i = 0; i < comm_size; i++)
        g_size = g_size + sizes[i];

      // Create the dataspace for the dataset.
      dimsf[0] = g_size;
      dimsf[1] = dat->dim;

      // Each process defines dataset in memory and writes it to a hyperslab
      // in the file.
      int disp = 0;
      for (int i = 0; i < my_rank; i++)
        disp = disp + sizes[i];
      count[0] = dat->set->size;
      count[1] = dimsf[1];
      offset[0] = disp;
      offset[1] = 0;
      memspace = H5Screate_simple(2, count, NULL);

      // Select hyperslab in the file.
      dataspace = H5Dget_space(dset_id);
      H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

      // Create property list for collective dataset write.
      H5Pclose(plist_id);
      plist_id = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

      // write data
      if (strcmp(dat->type, "double") == 0 ||
          strcmp(dat->type, "double:soa") == 0 ||
          strcmp(dat->type, "double precision") == 0 ||
          strcmp(dat->type, "real(8)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist_id,
                 dat->data);
      else if (strcmp(dat->type, "float") == 0 ||
               strcmp(dat->type, "float:soa") == 0 ||
               strcmp(dat->type, "real(4)") == 0 ||
               strcmp(dat->type, "real") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id,
                 dat->data);
      else if (strcmp(dat->type, "int") == 0 ||
               strcmp(dat->type, "int:soa") == 0 ||
               strcmp(dat->type, "int(4)") == 0 ||
               strcmp(dat->type, "integer") == 0 ||
               strcmp(dat->type, "integer(4)") == 0)
        H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id,
                 dat->data);
      else if ((strcmp(dat->type, "long") == 0) ||
               (strcmp(dat->type, "long:soa") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id,
                 dat->data);
      else if ((strcmp(dat->type, "long long") == 0) ||
               (strcmp(dat->type, "long long:soa") == 0))
        H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id,
                 dat->data);
      else {
        printf("Unknown type in op_fetch_data_hdf5_file()\n");
        MPI_Abort(OP_MPI_HDF5_WORLD, 2);
      }

      H5Dclose(dset_id);
      H5Pclose(plist_id);
      H5Sclose(memspace);
      H5Sclose(dataspace);
      H5Fclose(file_id);
      op_free(sizes);

      // free the temp op_dat used for this write
      op_free(dat->data);
      op_free(dat->set);
      op_free(dat);

      MPI_Comm_free(&OP_MPI_HDF5_WORLD);
      return;
    } else {
      op_printf("op_dat %s does not exists in the file ... creating data set\n",
                dat->name);
    }
  }

  //
  // new file and new data set ...
  //

  H5Pclose(plist_id);

  // find total size of dat
  int *sizes = (int *)xmalloc(sizeof(int) * comm_size);
  int g_size = 0;
  MPI_Allgather(&dat->set->size, 1, MPI_INT, sizes, 1, MPI_INT,
                OP_MPI_HDF5_WORLD);
  for (int i = 0; i < comm_size; i++)
    g_size = g_size + sizes[i];

  // Create the dataspace for the dataset.
  dimsf[0] = g_size;
  dimsf[1] = dat->dim;
  dataspace = H5Screate_simple(2, dimsf, NULL);

  // Each process defines dataset in memory and writes it to a hyperslab in the
  // file.
  int disp = 0;
  for (int i = 0; i < my_rank; i++)
    disp = disp + sizes[i];
  count[0] = dat->set->size;
  count[1] = dimsf[1];
  offset[0] = disp;
  offset[1] = 0;
  memspace = H5Screate_simple(2, count, NULL);

  // Select hyperslab in the file.
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, offset, NULL, count, NULL);

  // Create property list for collective dataset write.
  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  // Create the dataset with default properties and close dataspace.
  if (strcmp(dat->type, "double") == 0 ||
      strcmp(dat->type, "double:soa") == 0 ||
      strcmp(dat->type, "double precision") == 0 ||
      strcmp(dat->type, "real(8)") == 0) {
    dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_DOUBLE, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, memspace, dataspace, plist_id,
             dat->data);
  } else if (strcmp(dat->type, "float") == 0 ||
             strcmp(dat->type, "float:soa") == 0 ||
             strcmp(dat->type, "real(4)") == 0 ||
             strcmp(dat->type, "real") == 0) {
    dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, dataspace, plist_id,
             dat->data);
  } else if (strcmp(dat->type, "int") == 0 ||
             strcmp(dat->type, "int:soa") == 0 ||
             strcmp(dat->type, "int(4)") == 0 ||
             strcmp(dat->type, "integer") == 0 ||
             strcmp(dat->type, "integer(4)") == 0) {
    dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_INT, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_INT, memspace, dataspace, plist_id, dat->data);
  } else if ((strcmp(dat->type, "long") == 0) ||
             (strcmp(dat->type, "long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LONG, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_LONG, memspace, dataspace, plist_id,
             dat->data);
  } else if ((strcmp(dat->type, "long long") == 0) ||
             (strcmp(dat->type, "long long:soa") == 0)) {
    dset_id = H5Dcreate(file_id, dat->name, H5T_NATIVE_LLONG, dataspace,
                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset_id, H5T_NATIVE_LLONG, memspace, dataspace, plist_id,
             dat->data);
  } else {
    printf("Unknown type in op_fetch_data_hdf5_file()\n");
    MPI_Abort(OP_MPI_HDF5_WORLD, 2);
  }

  H5Dclose(dset_id);
  H5Pclose(plist_id);
  H5Sclose(memspace);
  H5Sclose(dataspace);
  op_free(sizes);

  /*attach attributes to dat*/

  // open existing data set
  dset_id = H5Dopen(file_id, dat->name, H5P_DEFAULT);
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

  // free the temp op_dat used for this write
  op_free(dat->data);
  op_free(dat->set);
  op_free(dat);

  MPI_Comm_free(&OP_MPI_HDF5_WORLD);
}
