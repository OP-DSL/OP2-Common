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

/*
 * op_util.h
 *
 * Header file for the utility functions used in op_util.c
 *
 * written by: Gihan R. Mudalige, (Started 01-03-2011)
 */

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
* MPI utility function prototypes
*******************************************************************************/

int compute_local_size(int global_size, int mpi_comm_size, int mpi_rank);

void *xmalloc(size_t size);

void *xcalloc(size_t number, size_t size);

void *xrealloc(void *ptr, size_t size);

int compare_sets(op_set set1, op_set set2);

int min(int array[], int size);

unsigned op2_hash(const char *s);

int binary_search(int a[], int value, int low, int high);

int linear_search(int a[], int value, int low, int high);

void quickSort(int arr[], int left, int right);

void quickSort_2(int arr1[], int arr2[], int left, int right);

void quickSort_dat(int arr[], char dat[], int left, int right, int elem_size);

void quickSort_map(int arr[], int map[], int left, int right, int dim);

int removeDups(int a[], int array_size);

int file_exist(char const *filename);

bool op_type_equivalence(const char *a, const char *b);

#ifdef __cplusplus
}
#endif

#endif /* __OP_UTIL_H */
