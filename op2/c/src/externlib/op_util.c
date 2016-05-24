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
 * op_mpi_util.c
 *
 * Some utility functions for the OP2 Distributed memory (MPI) implementation
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <op_lib_core.h>
#include <op_util.h>

/*******************************************************************************
* compute local size from global size
*******************************************************************************/

int compute_local_size (int global_size, int mpi_comm_size, int mpi_rank )
{
  int local_size = global_size/mpi_comm_size;
  int remainder = (int)fmod(global_size,mpi_comm_size);

  if (mpi_rank < remainder)
  {
    local_size = local_size + 1;
  }
  return local_size;
}

/*******************************************************************************
* Wrapper for malloc from www.gnu.org/
*******************************************************************************/

void* xmalloc (size_t size)
{
  if(size == 0) return (void *)NULL;

  register void *value = op_malloc (size);
  if (value == 0) printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for realloc from www.gnu.org/
*******************************************************************************/

void* xrealloc (void *ptr, size_t size)
{
  if(size == 0)
  {
    op_free(ptr);
    return (void *)NULL;
  }

  register void *value = op_realloc (ptr, size);
  if (value == 0) printf ("Virtual memory exhausted at realloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for calloc from www.gnu.org/
*******************************************************************************/

void* xcalloc (size_t number, size_t size)
{
  if(size == 0) return (void *)NULL;

  register void *value = op_calloc (number, size);
  if (value == 0) printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Return the index of the min value in an array
*******************************************************************************/

int min(int array[], int size)
{
  int min = 99; //initialized to 99 .. should check op_mpi_part_core and fix
  int index = -1;
  for(int i=0; i<size; i++)
  {
    if(array[i]<min)
    {
      index = i;
      min = array[i];
    }
  }
  return index;
}

/*******************************************************************************
* Binary search an array for a given value
*******************************************************************************/

int binary_search(int a[], int value, int low, int high)
{
  if (high < low)
    return -1; // not found
  else if (high == low)
    {
      if (a[low] == value) return low;
      else return -1;
    }
  else if (high == (low+1))
    {
      if (a[low] == value) return low;
      else if (a[high] == value) return high;
      else return -1;
    }

  int mid = low + (high - low) / 2;
  if (a[mid] > value)
    return binary_search(a, value, low, mid-1);
  else if (a[mid] < value)
    return binary_search(a, value, mid+1, high);
  else
    return mid; // found
}

/*******************************************************************************
* Linear search an array for a given value
*******************************************************************************/

int linear_search(int a[], int value, int low, int high)
{
  for(int i = low; i<=high; i++)
  {
    if (a[i] == value) return i;
  }
  return -1;
}

/*******************************************************************************
* Quicksort an array
*******************************************************************************/

void quickSort(int arr[], int left, int right)
{
  int i = left;
  int j = right;
  int tmp;
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    if (i <= j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
      i++; j--;
    }
  };
  // recursion
  if (left < j)
    quickSort(arr, left, j);
  if (i < right)
    quickSort(arr, i, right);
}

/*******************************************************************************
* Quick sort arr1 and organise arr2 elements according to the sorted arr1 order
*******************************************************************************/

void quickSort_2(int arr1[], int arr2[], int left, int right)
{
  int i = left;
  int j = right;
  int tmp1,tmp2;
  int pivot = arr1[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr1[i] < pivot)i++;
    while (arr1[j] > pivot)j--;
    if (i <= j) {
      tmp1 = arr1[i];
      arr1[i] = arr1[j];
      arr1[j] = tmp1;

      tmp2 = arr2[i];
      arr2[i] = arr2[j];
      arr2[j] = tmp2;
      i++; j--;
    }
  };
  // recursion
  if (left < j)
    quickSort_2(arr1, arr2, left, j);
  if (i < right)
    quickSort_2(arr1, arr2, i, right);
}

/*******************************************************************************
* Quick sort arr and organise dat[] elements according to the sorted arr order
*******************************************************************************/

void quickSort_dat(int arr[], char dat[], int left, int right, int elem_size)
{
  if(left < 0 || right <= 0)
    return;
  int i = left, j = right;
  int tmp;
  char* tmp_dat = (char *)xmalloc(sizeof(char)*elem_size);
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    //    if (i <= j) {
    if (i < j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;

      //tmp_dat = dat[i];
      memcpy(tmp_dat,(void *)&dat[i*elem_size],elem_size);
      //dat[i] = dat[j];
      memcpy(&dat[i*elem_size],(void *)&dat[j*elem_size],elem_size);
      //dat[j] = tmp_dat;
      memcpy(&dat[j*elem_size],(void *)tmp_dat,elem_size);
      i++; j--;
    }
    else if (i==j) {
      i++; j--;
    }
  };

  // recursion
  if (left < j)
    quickSort_dat(arr, dat, left, j, elem_size);
  if (i < right)
    quickSort_dat(arr, dat, i, right, elem_size);
  op_free(tmp_dat);
}

/*******************************************************************************
* Quick sort arr and organise map[] elements according to the sorted arr order
*******************************************************************************/

void quickSort_map(int arr[], int map[], int left, int right, int dim)
{
  int i = left, j = right;
  int tmp;
  int* tmp_map = (int *)xmalloc(sizeof(int)*dim);
  int pivot = arr[(left + right) / 2];

  // partition
  while (i <= j) {
    while (arr[i] < pivot)i++;
    while (arr[j] > pivot)j--;
    //    if (i <= j) {
    if (i < j) {
      tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;

      //tmp_dat = dat[i];
      memcpy(tmp_map,(void *)&map[i*dim],dim*sizeof(int));
      //dat[i] = dat[j];
      memcpy(&map[i*dim],(void *)&map[j*dim],dim*sizeof(int));
      //dat[j] = tmp_dat;
      memcpy(&map[j*dim],(void *)tmp_map,dim*sizeof(int));
      i++; j--;
    }
    else if (i==j) {
      i++; j--;
    }

  };

  // recursion
  if (left < j)
    quickSort_map(arr, map, left, j, dim);
  if (i < right)
    quickSort_map(arr, map, i, right, dim);
  op_free(tmp_map);
}

/*******************************************************************************
* Remove duplicates in an array
*******************************************************************************/

int removeDups(int a[], int array_size)
{
  int i, j;
  j = 0;
  // Remove the duplicates ...
  for (i = 1; i < array_size; i++)
  {
    if (a[i] != a[j])
    {
      j++;
      a[j] = a[i]; // Move it to the front
    }
  }
  // The new array size..
  array_size = (j + 1);
  return array_size;
}

/*******************************************************************************
* Check if a file exists
*******************************************************************************/
int file_exist (char const *filename)
{
  struct stat   buffer;
  return (stat (filename, &buffer) == 0);
}

const char *doubles[] = {"double","double:soa","real(8)","double precision"};
const char *floats[] = {"float","float:soa","real(4)","real"};
const char *ints[] = {"int","int:soa","integer(4)","integer"};

#ifdef __cplusplus
extern "C" {
#endif
bool op_type_equivalence(const char *a, const char *b) {
  for (int i = 0; i < 4; i++) {
    if (strcmp(a,doubles[i])==0) {      
      for (int j = 0; j < 4; j++) {
        if (strcmp(b,doubles[j])==0) {
          return true;
        }
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    if (strcmp(a,floats[i])==0) {      
      for (int j = 0; j < 4; j++) {
        if (strcmp(b,floats[j])==0) {
          return true;
        }
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    if (strcmp(a,ints[i])==0) {      
      for (int j = 0; j < 4; j++) {
        if (strcmp(b,ints[j])==0) {
          return true;
        }
      }
    }
  }
  return false;
}
#ifdef __cplusplus
}
#endif
