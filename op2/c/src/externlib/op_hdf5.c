/*
  Open source copyright declaration based on BSD open source template:
  http://www.opensource.org/licenses/bsd-license.php

* Copyright (c) 2009, Mike Giles
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
 * Implements the HDF5 based I/O routines for the OP2 single node
 * backe end
 *
 * written by: Gihan R. Mudalige, (Started 10-10-2011)
 */

#include <op_lib_c.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

// Use version 2 of H5Dopen H5Acreate and H5Dcreate
#define H5Dopen_vers 2
#define H5Acreate_vers 2
#define H5Dcreate_vers 2
//hdf5 header
#include <hdf5.h>

#include <op_hdf5.h>
#include <op_util.h> //just to include xmalloc routine

/*******************************************************************************
* Routine to write an op_set to an already open hdf5 file
*******************************************************************************/

op_set op_decl_set_hdf5(char const *file, char const *name)
{
  //HDF5 APIs definitions
  hid_t       file_id; //file identifier
  hid_t dset_id; //dataset identifier

  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);

  //Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);

  int l_size = 0;
  //read data
  H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &l_size);

  H5Dclose(dset_id);
  H5Fclose(file_id);

  return op_decl_set(l_size,  name);
}

/*******************************************************************************
* Routine to write an op_map to an already open hdf5 file
*******************************************************************************/

op_map op_decl_map_hdf5(op_set from, op_set to, int dim, char const *file, char const *name)
{
  //HDF5 APIs definitions
  hid_t       file_id; //file identifier
  hid_t dset_id; //dataset identifier
  hid_t       dataspace; //data space identifier

  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT );


  /*find total size of this map by reading attributes*/
  int g_size;
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  hid_t attr = H5Aopen(dset_id, "size", H5P_DEFAULT);
  //read attribute
  H5Aread(attr,H5T_NATIVE_INT,&g_size);
  H5Aclose(attr);
  H5Dclose(dset_id);

  //check if size is accurate
  if(from->size != g_size)
  {
    printf("map from set size %d in file %s and size %d do not match \n",
        g_size,file,from->size);
    exit(2);
  }


  /*find dim with available attributes*/
  int map_dim = 0;
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  //read attribute
  H5Aread(attr,H5T_NATIVE_INT,&map_dim);
  H5Aclose(attr);
  H5Dclose(dset_id);
  if(map_dim != dim)
  {
    printf("map.dim %d in file %s and dim %d do not match\n",map_dim,file,dim);
    exit(2);
  }

  /*find type with available attributes*/
  dataspace= H5Screate(H5S_SCALAR);
  hid_t  atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, 10);
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);
  //read attribute
  char typ[10];
  H5Aread(attr,atype,typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  H5Dclose(dset_id);

  //Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  //initialize data buffer and read data
  int* map;
  if(strcmp(typ,"int") == 0)
  {
    map = (int *)xmalloc(sizeof(int)*g_size*dim);
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  }
  else if (strcmp(typ,"long") == 0)
  {
    map = (int *)xmalloc(sizeof(long)*g_size*dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  }
  else if (strcmp(typ,"long long") == 0)
  {
    map = (int *)xmalloc(sizeof(long long)*g_size*dim);
    H5Dread(dset_id, H5T_NATIVE_LONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, map);
  }
  else
  {
    printf("Unknown type in file %s for map %s\n",file, name);
    exit(2);
  }

  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);

  return op_decl_map(from, to, dim, map, name);
}


/*******************************************************************************
* Routine to write an op_map to an already open hdf5 file
*******************************************************************************/

op_dat op_decl_dat_hdf5(op_set set, int dim, char const *type, char const *file, char const *name)
{
  //HDF5 APIs definitions
  hid_t       file_id; //file identifier
  hid_t dset_id; //dataset identifier
  hid_t       dataspace; //data space identifier
  hid_t attr;   //attribute identifier

  file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);

  /*find element size of this dat with available attributes*/
  size_t dat_size = 0;
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  attr = H5Aopen(dset_id, "size", H5P_DEFAULT);
  //read attribute
  H5Aread(attr,H5T_NATIVE_INT,&dat_size);
  H5Aclose(attr);
  H5Dclose(dset_id);

  /*find dim with available attributes*/
  int dat_dim = 0;
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  attr = H5Aopen(dset_id, "dim", H5P_DEFAULT);
  //read attribute
  H5Aread(attr,H5T_NATIVE_INT,&dat_dim);
  H5Aclose(attr);
  H5Dclose(dset_id);
  if(dat_dim != dim)
  {
    printf("dat.dim %d in file %s and dim %d do not match\n",dat_dim,file,dim);
    exit(2);
  }

  /*find type with available attributes*/
  dataspace= H5Screate(H5S_SCALAR);
  hid_t  atype = H5Tcopy(H5T_C_S1);
  H5Tset_size(atype, 10);
  //open existing data set
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  //get OID of the attribute
  attr = H5Aopen(dset_id, "type", H5P_DEFAULT);
  //read attribute
  char typ[10];
  H5Aread(attr,atype,typ);
  H5Aclose(attr);
  H5Sclose(dataspace);
  H5Dclose(dset_id);
  if(strcmp(typ,type) != 0)
  {
    printf("dat.type %s in file %s and type %s do not match\n",typ,file,type);
    exit(2);
  }

  //Create the dataset with default properties and close dataspace.
  dset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  dataspace = H5Dget_space(dset_id);

  //initialize data buffer and read in data
  char* data;
  if(strcmp(type,"double") == 0)
  {
    data = (char *)xmalloc(set->size*dim*sizeof(double));
    H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    if(dat_size != dim*sizeof(double))
    {
      printf("dat.size %lu in file %s and %d*sizeof(double) do not match\n",dat_size,file,dim);
      exit(2);
    }
    else
      dat_size = sizeof(double);

  }else if(strcmp(type,"float") == 0)
  {
    data = (char *)xmalloc(set->size*dim*sizeof(float));
    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    if(dat_size != dim*sizeof(float))
    {
      printf("dat.size %lu in file %s and %d*sizeof(float) do not match\n",dat_size,file,dim);
      exit(2);
    }
    else
      dat_size = sizeof(float);

  }else if(strcmp(type,"int") == 0)
  {
    data = (char *)xmalloc(set->size*dim*sizeof(int));
    H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    if(dat_size != dim*sizeof(int))
    {
      printf("dat.size %lu in file %s and %d*sizeof(int) do not match\n",dat_size,file,dim);
      exit(2);
    }
    else
      dat_size = sizeof(int);
  }else
  {
    printf("unknown type\n");
    exit(2);
  }

  H5Sclose(dataspace);
  H5Dclose(dset_id);
  H5Fclose(file_id);

  return op_decl_dat(set, dim, type, dat_size, data, name );
}

/*******************************************************************************
* Routine to write all to a named hdf5 file
*******************************************************************************/

void op_write_hdf5(char const * file_name)
{
  (void)file_name;
}

