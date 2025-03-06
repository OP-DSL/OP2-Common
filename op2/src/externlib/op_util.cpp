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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>

#include <op_lib_core.h>
#include <op_util.h>

/*******************************************************************************
* compute local size from global size
*******************************************************************************/

int compute_local_size(int global_size, int mpi_comm_size, int mpi_rank) {
  int local_size = global_size / mpi_comm_size;
  int remainder = (int)fmod(global_size, mpi_comm_size);

  if (mpi_rank < remainder) {
    local_size = local_size + 1;
  }
  return local_size;
}

/*******************************************************************************
* Wrapper for malloc from www.gnu.org/
*******************************************************************************/

void *xmalloc(size_t size) {
  if (size == 0)
    return (void *)NULL;

  void *value = op_malloc(size);
  if (value == 0)
    printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for realloc from www.gnu.org/
*******************************************************************************/

void *xrealloc(void *ptr, size_t size) {
  if (size == 0) {
    op_free(ptr);
    return (void *)NULL;
  }

  void *value = op_realloc(ptr, size);
  if (value == 0)
    printf("Virtual memory exhausted at realloc\n");
  return value;
}

/*******************************************************************************
* Wrapper for calloc from www.gnu.org/
*******************************************************************************/

void *xcalloc(size_t number, size_t size) {
  if (size == 0)
    return (void *)NULL;

  void *value = op_calloc(number, size);
  if (value == 0)
    printf("Virtual memory exhausted at malloc\n");
  return value;
}

/*******************************************************************************
* Return the index of the min value in an array
*******************************************************************************/

int min(int array[], int size) {
  int min = 99; // initialized to 99 .. should check op_mpi_part_core and fix
  int index = -1;
  for (int i = 0; i < size; i++) {
    if (array[i] < min) {
      index = i;
      min = array[i];
    }
  }
  return index;
}

/*******************************************************************************
* Binary search an array for a given value
*******************************************************************************/

template <typename T, typename U>
idx_l_t binary_search(T a[], U value, int low, int high) {
  auto lb = std::lower_bound(a + low, a + high + 1, value);
  if (lb == a + high + 1 || *lb != value) return -1;
  return (int) (lb - a);
}

template idx_l_t binary_search<idx_g_t, idx_g_t>(idx_g_t a[], idx_g_t value, int low, int high);
template idx_l_t binary_search<idx_g_t, idx_l_t>(idx_g_t a[], idx_l_t value, int low, int high);
template idx_l_t binary_search<idx_l_t, idx_l_t>(idx_l_t a[], idx_l_t value, int low, int high);

/*******************************************************************************
* Linear search an array for a given value
*******************************************************************************/

int linear_search(int a[], int value, int low, int high) {
  for (int i = low; i <= high; i++) {
    if (a[i] == value)
      return i;
  }
  return -1;
}

/*******************************************************************************
* Quick sort arr1 and organise arr2 elements according to the sorted arr1 order
*******************************************************************************/

void op_sort_2(int *__restrict xs, int *__restrict ys, int n) {
  std::sort(ZipIter(xs, ys), ZipIter(xs + n, ys + n));
}

/*******************************************************************************
* Quick sort arr and organise dat[] elements according to the sorted arr order
*******************************************************************************/

void op_sort_dat(idx_g_t *__restrict arr, char *__restrict dat, int n, int elem_size2) {
  size_t elem_size = elem_size2;
  idx_g_t *indicies = (idx_g_t *) xmalloc(n * sizeof(idx_g_t));

  for (idx_g_t i = 0; i < n; ++i)
    indicies[i] = i;

  std::sort(ZipIter(arr, indicies), ZipIter(arr + n, indicies + n));
 
  char *tmp_dat = (char *) xmalloc(n * elem_size * sizeof(char));
  for (idx_g_t i = 0; i < n; ++i)
    std::copy(dat + indicies[i] * elem_size, dat + (indicies[i] + 1) * elem_size, tmp_dat + i * elem_size);

  std::copy(tmp_dat, tmp_dat + n * elem_size, dat);

  op_free(tmp_dat);
  op_free(indicies);

  return;
}

/*******************************************************************************
* Quick sort arr and organise map[] elements according to the sorted arr order
*******************************************************************************/

void op_sort_map(idx_g_t *__restrict arr, idx_g_t *__restrict map, int n, int dim) {
  op_sort_dat(arr, (char *) map, n, dim * sizeof(idx_g_t));
}

/*******************************************************************************
* Check if a file exists
*******************************************************************************/
int file_exist(char const *filename) {
  struct stat buffer;
  return (stat(filename, &buffer) == 0);
}

const char *doubles[] = {"double", "double:soa", "real(8)", "double precision"};
const char *floats[] = {"float", "float:soa", "real(4)", "real"};
const char *ints[] = {"int", "int:soa", "integer(4)", "integer"};

#ifdef __cplusplus
extern "C" {
#endif
bool op_type_equivalence(const char *a, const char *b) {
  for (int i = 0; i < 4; i++) {
    if (strcmp(a, doubles[i]) == 0) {
      for (int j = 0; j < 4; j++) {
        if (strcmp(b, doubles[j]) == 0) {
          return true;
        }
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    if (strcmp(a, floats[i]) == 0) {
      for (int j = 0; j < 4; j++) {
        if (strcmp(b, floats[j]) == 0) {
          return true;
        }
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    if (strcmp(a, ints[i]) == 0) {
      for (int j = 0; j < 4; j++) {
        if (strcmp(b, ints[j]) == 0) {
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
