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

#ifndef __OP_UTIL_H
#define __OP_UTIL_H

#include <tuple>
#include <array>
#include <algorithm>

/*
 * op_util.h
 *
 * Header file for the utility functions used in op_util.c
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

/*******************************************************************************
* MPI utility function prototypes
*******************************************************************************/

template <typename T, typename U>
int binary_search(T a[], U value, int low, int high); 

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
  auto val() const { return *std::get<0>(pointers); };

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

#ifdef __cplusplus
extern "C" {
#endif

int compute_local_size(int global_size, int mpi_comm_size, int mpi_rank);

void *xmalloc(size_t size);

void *xcalloc(size_t number, size_t size);

void *xrealloc(void *ptr, size_t size);

int compare_sets(op_set set1, op_set set2);

int min(int array[], int size);

unsigned op2_hash(const char *s);

int linear_search(int a[], int value, int low, int high);

void op_sort_2(int *__restrict arr1, int *__restrict arr2, int n);

void op_sort_dat(idx_g_t *__restrict arr, char *__restrict dat, int n, int elem_size);

void op_sort_map(idx_g_t *__restrict arr, idx_g_t *__restrict map, int n, int dim);

int file_exist(char const *filename);

bool op_type_equivalence(const char *a, const char *b);

#ifdef __cplusplus
}
#endif

// Sort an array
template <typename T>
void op_sort(T *__restrict arr, int n) {
  std::sort(arr, arr + n);
}

// Remove duplicates in an array, assumes the array is sorted
template <typename T>
int removeDups(T *__restrict arr, int n) {
  return std::unique(arr, arr + n) - arr;
}

#endif /* __OP_UTIL_H */
