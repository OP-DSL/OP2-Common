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
#include <tuple>
#include <array>

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

  register void *value = op_malloc(size);
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

  register void *value = op_realloc(ptr, size);
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

  register void *value = op_calloc(number, size);
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
int binary_search(int a[], int value, int low, int high) {
  auto lb = std::lower_bound(a + low, a + high + 1, value);
  if (lb == a + high + 1 || *lb != value) return -1;
  return (int) (lb - a);
}

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
* Quicksort an array
*******************************************************************************/

void op_sort(int *__restrict xs, int n) {
  std::sort(xs, xs + n);
}

/*******************************************************************************
* Quick sort arr1 and organise arr2 elements according to the sorted arr1 order
*******************************************************************************/

// C++23 zip range view at home
// adapted from https://github.com/dpellegr/ZipIterator
template<typename... Ts>
struct ZipRef {
  std::tuple<Ts*...> pointers;

  ZipRef() = delete;
  ZipRef(Ts*... pointers): pointers(pointers...) {};

  template <std::size_t I = 0>
  void copy_assign(const ZipRef& rhs) {
    *(std::get<I>(pointers)) = *(std::get<I>(rhs.pointers));
    if constexpr(I + 1 < sizeof...(Ts) ) copy_assign<I + 1>(rhs);
  }

  template <std::size_t I = 0>
  void val_assign(const std::tuple<Ts...>& rhs) {
    *(std::get<I>(pointers)) = std::get<I>(rhs);
    if constexpr(I + 1 < sizeof...(Ts)) val_assign<I + 1>(rhs);
  }

  ZipRef& operator=(const ZipRef& rhs) { copy_assign(rhs); return *this; };
  ZipRef& operator=(const std::tuple<Ts...>& rhs) { val_assign(rhs); return *this; };

  operator std::tuple<Ts...>() const { return std::apply([](auto&&... p) { return std::tuple((*p)...); }, pointers); }
  int val() const { return *std::get<0>(pointers); };

  #define OPERATOR(OP) \
    bool operator OP(const ZipRef& rhs) const { return val() OP rhs.val(); } \
    inline friend bool operator OP(const ZipRef& r, const std::tuple<Ts...>& t) { return r.val() OP std::get<0>(t); } \
    inline friend bool operator OP(const std::tuple<Ts...>& t, const ZipRef& r) { return std::get<0>(t) OP r.val(); }

    OPERATOR(==) OPERATOR(<=) OPERATOR(>=)
    OPERATOR(!=) OPERATOR(<)  OPERATOR(>)
  #undef OPERATOR
};

template<typename std::size_t I = 0, typename... Ts>
void swap(const ZipRef<Ts...>& z1, const ZipRef<Ts...>& z2) {
  std::swap(*std::get<I>(z1.pointers), *std::get<I>(z2.pointers));
  if constexpr(I + 1 < sizeof...(Ts)) swap<I + 1, Ts...>(z1, z2);
}

template<typename... Ts>
struct ZipIter {
  std::tuple<Ts...> iterators;

  using iterator_category = std::common_type_t<typename std::iterator_traits<Ts>::iterator_category...>;
  using difference_type   = std::common_type_t<typename std::iterator_traits<Ts>::difference_type...>;
  using value_type        = std::tuple<typename std::iterator_traits<Ts>::value_type...>;
  using pointer           = std::tuple<typename std::iterator_traits<Ts>::pointer...>;
  using reference         = ZipRef<std::remove_reference_t<typename std::iterator_traits<Ts>::reference>...>;

  ZipIter() = delete;
  ZipIter(Ts... iterators): iterators(iterators...) {};

  ZipIter& operator+=(const difference_type d) { std::apply([&](auto& ...it) { (std::advance(it, d), ...); }, iterators); return *this; }

  ZipIter& operator-=(const difference_type d) { return operator+=(-d); }

  reference operator* () const {return std::apply([](auto&&... it){ return reference(&(*(it))...); }, iterators);}

  ZipIter& operator++() { return operator+=( 1); }
  ZipIter& operator--() { return operator+=(-1); }
  ZipIter operator++(int) { ZipIter tmp(*this); operator++(); return tmp; }
  ZipIter operator--(int) { ZipIter tmp(*this); operator--(); return tmp; }

  difference_type operator-(const ZipIter& rhs) const { return std::get<0>(iterators) - std::get<0>(rhs.iterators); }
  ZipIter operator+(const difference_type d) const { ZipIter tmp(*this); tmp += d; return tmp; }
  ZipIter operator-(const difference_type d) const { ZipIter tmp(*this); tmp -= d; return tmp; }
  inline friend ZipIter operator+(const difference_type d, const ZipIter& z) { return z+d; }
  inline friend ZipIter operator-(const difference_type d, const ZipIter& z) { return z - d; }

  bool operator==(const ZipIter& rhs) const { return iterators == rhs.iterators; }
  bool operator!=(const ZipIter& rhs) const { return iterators != rhs.iterators; }

  #define OPERATOR(OP) \
    bool operator OP(const ZipIter& rhs) const { return std::get<0>(iterators) OP std::get<0>(rhs.iterators); }
    OPERATOR(<=) OPERATOR(>=)
    OPERATOR(<)  OPERATOR(>)
  #undef OPERATOR
};

void op_sort_2(int *__restrict xs, int *__restrict ys, int n) {
  std::sort(ZipIter(xs, ys), ZipIter(xs + n, ys + n));
}

/*******************************************************************************
* Quick sort arr and organise dat[] elements according to the sorted arr order
*******************************************************************************/

void op_sort_dat(int *__restrict arr, char *__restrict dat, int n, int elem_size2) {
  size_t elem_size = elem_size2;
  int *indicies = (int *) xmalloc(n * sizeof(int));

  for (int i = 0; i < n; ++i)
    indicies[i] = i;

  std::sort(ZipIter(arr, indicies), ZipIter(arr + n, indicies + n));
 
  char *tmp_dat = (char *) xmalloc(n * elem_size * sizeof(char));
  for (int i = 0; i < n; ++i)
    std::copy(dat + indicies[i] * elem_size, dat + (indicies[i] + 1) * elem_size, tmp_dat + i * elem_size);

  std::copy(tmp_dat, tmp_dat + n * elem_size, dat);

  op_free(tmp_dat);
  op_free(indicies);

  return;
}

/*******************************************************************************
* Quick sort arr and organise map[] elements according to the sorted arr order
*******************************************************************************/

void op_sort_map(int *__restrict arr, int *__restrict map, int n, int dim) {
  op_sort_dat(arr, (char *) map, n, dim * sizeof(int));
}

/*******************************************************************************
* Remove duplicates in an array
*******************************************************************************/

int removeDups(int a[], int array_size) {
  int i, j;
  j = 0;
  // Remove the duplicates ...
  for (i = 1; i < array_size; i++) {
    if (a[i] != a[j]) {
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
