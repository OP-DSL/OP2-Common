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

#include <mpi.h>

#include <op_lib_c.h>
#include <op_lib_mpi.h>
#include <op_util.h>
#include <extern/rapidhash.h>
#include <op_lib_mpi_unified.hpp>
#include <set>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

MPI_Comm OP_MPI_IO_WORLD;

void _mpi_allgather(int *l, int *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, comm);
}

void _mpi_allgather(float *l, float *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, comm);
}

void _mpi_allgather(double *l, double *g, int size, int *recevcnts, int *displs,
                    MPI_Comm comm) {
  MPI_Allgatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, comm);
}

void _mpi_gather(int *l, int *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_INT, g, recevcnts, displs, MPI_INT, MPI_ROOT, comm);
}

void _mpi_gather(float *l, float *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_FLOAT, g, recevcnts, displs, MPI_FLOAT, MPI_ROOT,
              comm);
}

void _mpi_gather(double *l, double *g, int size, int *recevcnts, int *displs,
                 MPI_Comm comm) {
  MPI_Gatherv(l, size, MPI_DOUBLE, g, recevcnts, displs, MPI_DOUBLE, MPI_ROOT,
              comm);
}

template <typename T>
void gather_data_hdf5(op_dat dat, char *usr_ptr, int low, int high) {
  // create new communicator
  int my_rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &my_rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  // compute local number of elements in dat
  int count = dat->set->size;

  T *l_array = (T *)xmalloc(dat->dim * (count) * sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]), (size_t)dat->size * count);
  int l_size = count;
  size_t elem_size = dat->dim;
  int *recevcnts = (int *)xmalloc(comm_size * sizeof(int));
  int *displs = (int *)xmalloc(comm_size * sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for (int i = 0; i < comm_size; i++) {
    g_size += recevcnts[i];
    recevcnts[i] = elem_size * recevcnts[i];
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + recevcnts[i];
  }

  g_array = (T *)xmalloc(elem_size * g_size * sizeof(T));

  // need to all-gather dat->data and copy this to the memory block pointed by
  // usr_ptr
  _mpi_allgather(l_array, g_array, l_size * elem_size, recevcnts, displs,
                 OP_MPI_IO_WORLD);

  if (low < 0 || high > g_size - 1) {
    printf("op_fetch_data: Indices not within range of elements held in %s\n",
           dat->name);
    MPI_Abort(OP_MPI_IO_WORLD, -1);
  }
  memcpy((void *)usr_ptr, (void *)&g_array[low * (size_t)dat->size],
         (high + 1) * (size_t)dat->size);

  free(l_array);
  free(recevcnts);
  free(displs);
  free(g_array);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

void checked_write(int v, const char *file_name) {
  if (v) {
    printf("error writing to %s\n", file_name);
    MPI_Abort(OP_MPI_IO_WORLD, -1);
  }
}

template <typename T>
void write_bin(FILE *fp, int g_size, int elem_size, T *g_array,
               const char *file_name) {
  checked_write(fwrite(&g_size, sizeof(int), 1, fp) < 1, file_name);
  checked_write(fwrite(&elem_size, sizeof(int), 1, fp) < 1, file_name);

  for (int i = 0; i < g_size; i++)
    checked_write(fwrite(&g_array[i * elem_size], sizeof(T), elem_size, fp) <
                      (size_t)elem_size,
                  file_name);
}

template <typename T, const char *fmt>
void write_txt(FILE *fp, int g_size, int elem_size, T *g_array,
               const char *file_name) {
  checked_write(fprintf(fp, "%d %d\n", g_size, elem_size) < 0, file_name);

  for (int i = 0; i < g_size; i++) {
    for (int j = 0; j < elem_size; j++)
      checked_write(fprintf(fp, fmt, g_array[i * elem_size + j]) < 0,
                    file_name);
    fprintf(fp, "\n");
  }
}

template <typename T, void (*F)(FILE *, int, int, T *, const char *)>
void write_file(op_dat dat, const char *file_name) {
  // create new communicator for output
  int rank, comm_size;
  MPI_Comm_dup(OP_MPI_WORLD, &OP_MPI_IO_WORLD);
  MPI_Comm_rank(OP_MPI_IO_WORLD, &rank);
  MPI_Comm_size(OP_MPI_IO_WORLD, &comm_size);

  // compute local number of elements in dat
  int count = dat->set->size;

  T *l_array = (T *)xmalloc(dat->dim * (count) * sizeof(T));
  memcpy(l_array, (void *)&(dat->data[0]), (size_t)dat->size * count);

  int l_size = count;
  int elem_size = dat->dim;
  int *recevcnts = (int *)xmalloc(comm_size * sizeof(int));
  int *displs = (int *)xmalloc(comm_size * sizeof(int));
  int disp = 0;
  T *g_array = 0;

  MPI_Allgather(&l_size, 1, MPI_INT, recevcnts, 1, MPI_INT, OP_MPI_IO_WORLD);

  int g_size = 0;
  for (int i = 0; i < comm_size; i++) {
    g_size += recevcnts[i];
    recevcnts[i] = elem_size * recevcnts[i];
  }
  for (int i = 0; i < comm_size; i++) {
    displs[i] = disp;
    disp = disp + recevcnts[i];
  }
  if (rank == MPI_ROOT)
    g_array = (T *)xmalloc(elem_size * g_size * sizeof(T));
  _mpi_gather(l_array, g_array, l_size * elem_size, recevcnts, displs,
              OP_MPI_IO_WORLD);

  if (rank == MPI_ROOT) {
    FILE *fp;
    if ((fp = fopen(file_name, "w")) == NULL) {
      printf("can't open file %s\n", file_name);
      MPI_Abort(OP_MPI_IO_WORLD, -1);
    }

    // Write binary or text as requested by the caller
    F(fp, g_size, elem_size, g_array, file_name);

    fclose(fp);
    free(g_array);
  }

  free(l_array);
  free(recevcnts);
  free(displs);
  MPI_Comm_free(&OP_MPI_IO_WORLD);
}

/*******************************************************************************
* Routine to fetch data from an op_dat to user allocated memory block under hdf5
* -- placed in op_mpi_core.c as this routine does not use any hdf5 functions
*******************************************************************************/

void fetch_data_hdf5(op_dat dat, char *usr_ptr, int low, int high) {
  if (strcmp(dat->type, "double") == 0)
    gather_data_hdf5<double>(dat, usr_ptr, low, high);
  else if (strcmp(dat->type, "float") == 0)
    gather_data_hdf5<float>(dat, usr_ptr, low, high);
  else if (strcmp(dat->type, "int") == 0)
    gather_data_hdf5<int>(dat, usr_ptr, low, high);
  else
    printf("Unknown type %s, cannot error in fetch_data_hdf5() \n", dat->type);
}

/*******************************************************************************
 * Write a op_dat to a named ASCI file
 *******************************************************************************/

extern const char fmt_double[] = "%f ";
extern const char fmt_float[] = "%f ";
extern const char fmt_int[] = "%d ";

void print_dat_to_txtfile_mpi(op_dat dat, const char *file_name) {
  if (strcmp(dat->type, "double") == 0)
    write_file<double, write_txt<double, fmt_double> >(dat, file_name);
  else if (strcmp(dat->type, "float") == 0)
    write_file<float, write_txt<float, fmt_float> >(dat, file_name);
  else if (strcmp(dat->type, "int") == 0)
    write_file<int, write_txt<int, fmt_int> >(dat, file_name);
  else
    printf("Unknown type %s, cannot be written to file %s\n", dat->type,
           file_name);
}

/*******************************************************************************
 * Write a op_dat to a named Binary file
 *******************************************************************************/

void print_dat_to_binfile_mpi(op_dat dat, const char *file_name) {
  if (strcmp(dat->type, "double") == 0)
    write_file<double, write_bin<double> >(dat, file_name);
  else if (strcmp(dat->type, "float") == 0)
    write_file<float, write_bin<float> >(dat, file_name);
  else if (strcmp(dat->type, "int") == 0)
    write_file<int, write_bin<int> >(dat, file_name);
  else
    printf("Unknown type %s, cannot be written to file %s\n", dat->type,
           file_name);
}

void gather_data_to_buffer_ptr_cuda(op_arg arg, halo_list eel, halo_list enl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<size_t>& neigh_offsets);
void scatter_data_from_buffer_ptr_cuda(op_arg arg, halo_list iel, halo_list inl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<size_t>& neigh_offsets);

void gather_data_to_buffer_ptr(op_arg arg, halo_list eel, halo_list enl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<size_t>& neigh_offsets) {

  for (int i = 0; i < eel->ranks_size; i++) {
    int dest_rank = eel->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    unsigned buf_pos = neigh_offsets[buf_rankpos];
    for (int j = 0; j < eel->sizes[i]; j++) {
      unsigned set_elem_index = eel->list[eel->disps[i] + j];
      memcpy(&buffer[buf_pos + j * (size_t)arg.dat->size],
               (void *)&arg.dat->data[(size_t)arg.dat->size * (set_elem_index)], arg.dat->size);
    }
    neigh_offsets[buf_rankpos] += eel->sizes[i] * (size_t)arg.dat->size;
  }
  for (int i = 0; i < enl->ranks_size; i++) {
    int dest_rank = enl->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    unsigned buf_pos = neigh_offsets[buf_rankpos];
    for (int j = 0; j < enl->sizes[i]; j++) {
      unsigned set_elem_index = enl->list[enl->disps[i] + j];
      memcpy(&buffer[buf_pos + j * (size_t)arg.dat->size],
               (void *)&arg.dat->data[(size_t)arg.dat->size * (set_elem_index)], arg.dat->size);
    }
    neigh_offsets[buf_rankpos] += enl->sizes[i] * (size_t)arg.dat->size;
  }
}

void scatter_data_from_buffer_ptr(op_arg arg, halo_list iel, halo_list inl, char *buffer, 
                               std::vector<int>& neigh_list, std::vector<size_t>& neigh_offsets) {

  for (int i = 0; i < iel->ranks_size; i++) {
    int dest_rank = iel->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    unsigned buf_pos = neigh_offsets[buf_rankpos];
    for (int j = 0; j < iel->sizes[i]; j++) {
      // if (*(double*)&arg.dat->data[arg.dat->size * (arg.dat->set->size + iel->disps[i] + j)] !=
      //     *(double*)&buffer[buf_pos + j * arg.dat->size])
      //     printf("Mismatch\n");
      memcpy((void *)&arg.dat->data[(size_t)arg.dat->size * (arg.dat->set->size + iel->disps[i] + j)], 
              &buffer[buf_pos + j * (size_t)arg.dat->size], arg.dat->size);
    }
    neigh_offsets[buf_rankpos] += iel->sizes[i] * (size_t)arg.dat->size;
  }
  for (int i = 0; i < inl->ranks_size; i++) {
    int dest_rank = inl->ranks[i];
    int buf_rankpos = std::distance(neigh_list.begin(),std::lower_bound(neigh_list.begin(), neigh_list.end(), dest_rank));
    unsigned buf_pos = neigh_offsets[buf_rankpos];
    for (int j = 0; j < inl->sizes[i]; j++) {
      // if (*(double*)&arg.dat->data[arg.dat->size * (arg.dat->set->size + iel->size + inl->disps[i] + j)] !=
      //     *(double*)&buffer[buf_pos + j * arg.dat->size])
      //     printf("Mismatch2\n");
      memcpy((void *)&arg.dat->data[(size_t)arg.dat->size * (arg.dat->set->size + iel->size + inl->disps[i] + j)], 
              &buffer[buf_pos + j * (size_t)arg.dat->size], (size_t)arg.dat->size);
    }
    neigh_offsets[buf_rankpos] += inl->sizes[i] * (size_t)arg.dat->size;
  }
}

constexpr uint64_t hash_seed_default = RAPID_SEED;

template<typename T>
static inline uint64_t hash(const T key, uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed((void *)&key, sizeof(T), seed);
}

template<typename T>
static inline uint64_t hash(const T* key, size_t len, uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed((void *)key, sizeof(T) * len, seed);
}

template<>
inline uint64_t hash(const void* key, size_t len, uint64_t seed) {
    return rapidhash_withSeed(key, len, seed);
}

struct ExchangeInfo {
    // MPI_Comm comm;

    std::vector<int> dats;
    std::map<int, int> partial_exchanges;

    std::vector<int> send_neighbours;
    std::vector<int> recv_neighbours;

    std::vector<int> send_sizes;
    std::vector<int> recv_sizes;

    std::vector<int> send_offsets;
    std::vector<int> recv_offsets;

    PtrBuffers ptrs;
};

std::unordered_map<uint64_t, ExchangeInfo> exchanges;

ExchangeInfo *active_exchange_4 = nullptr;
ExchangeInfo *active_exchange_8 = nullptr;

static op_dat get_dat(int index) {
    op_dat_entry *item;
    TAILQ_FOREACH(item, &OP_dat_list, entries) {
        if (item->dat->index == index) return item->dat;
    }

    assert(false);
}

static std::set<int> get_neighbours(halo_list list) {
    std::set<int> neighbours;
    for (int i = 0; i < list->ranks_size; ++i) {
        neighbours.insert(list->ranks[i]);
    }

    return neighbours;
}

static void extract_indirect_lists(halo_list hlist, op_dat dat,
                                   std::unordered_map<int, std::vector<void *>> &lists) {
    auto elem_size = dat->size / dat->dim;
    auto stride = round32(dat->set->size + OP_import_exec_list[dat->set->index]->size
                                         + OP_import_nonexec_list[dat->set->index]->size);

    for (int i = 0; i < hlist->ranks_size; ++i) {
        auto neighbour = hlist->ranks[i];
        auto num_elem = hlist->sizes[i];
        auto list_disp = hlist->disps[i];

        auto& list = lists[neighbour];
        for (int j = list_disp; j < list_disp + num_elem; ++j) {
            auto index = hlist->list[j];

            for (int d = 0; d < dat->dim; ++d) {
                list.push_back(&dat->data_d[elem_size * (index + d * stride)]);
            }
        }
    }
}

static void extract_blocked_lists(halo_list hlist, op_dat dat, int offset,
                                  std::unordered_map<int, std::vector<void *>> &lists) {
    auto elem_size = dat->size / dat->dim;
    auto stride = round32(dat->set->size + OP_import_exec_list[dat->set->index]->size
                                         + OP_import_nonexec_list[dat->set->index]->size);

    for (int i = 0; i < hlist->ranks_size; ++i) {
        auto neighbour = hlist->ranks[i];
        auto num_elem = hlist->sizes[i];
        auto list_disp = hlist->disps[i];

        auto& list = lists[neighbour];
        for (int j = list_disp; j < list_disp + num_elem; ++j) {
            for (int d = 0; d < dat->dim; ++d) {
                list.push_back(&dat->data_d[elem_size * (offset + j + d * stride)]);
            }
        }
    }
}

// Pre: dats have same element size
static void init_exchange_info(const std::set<int> &dats,
                               const std::map<int, int> &partial_exchanges,
                               ExchangeInfo& exchange_info) {
    exchange_info.dats = std::vector<int>(dats.begin(), dats.end());
    exchange_info.partial_exchanges = partial_exchanges;

    std::set<int> full_exchange_sets;
    for (auto dat_index : dats) {
        if (partial_exchanges.find(dat_index) != partial_exchanges.end()) {
            continue;
        }

        full_exchange_sets.insert(get_dat(dat_index)->set->index);
    }

    std::set<int> send_neighbours;
    std::set<int> recv_neighbours;

    for (auto set : full_exchange_sets) {
        send_neighbours.merge(get_neighbours(OP_export_exec_list[set]));
        send_neighbours.merge(get_neighbours(OP_export_nonexec_list[set]));

        recv_neighbours.merge(get_neighbours(OP_import_exec_list[set]));
        recv_neighbours.merge(get_neighbours(OP_import_nonexec_list[set]));
    }

    for (auto [dat_index, map_index] : partial_exchanges) {
        send_neighbours.merge(get_neighbours(OP_export_nonexec_permap[map_index]));
        recv_neighbours.merge(get_neighbours(OP_import_nonexec_permap[map_index]));
    }

    exchange_info.send_neighbours = std::vector(send_neighbours.begin(), send_neighbours.end());
    exchange_info.recv_neighbours = std::vector(recv_neighbours.begin(), recv_neighbours.end());

    /*
    auto err = MPI_Dist_graph_create_adjacent(
            OP_MPI_WORLD,
            exchange_info.recv_neighbours.size(),
            exchange_info.recv_neighbours.data(),
            MPI_UNWEIGHTED,
            exchange_info.send_neighbours.size(),
            exchange_info.send_neighbours.data(),
            MPI_UNWEIGHTED,
            MPI_INFO_NULL,
            false,
            &exchange_info.comm
    );

    if (err != MPI_SUCCESS) {
        printf("Error: could not create graph communicator: %d\n", err);
        exit(1);
    }
    */

    std::unordered_map<int, std::vector<void *>> gather_lists;
    std::unordered_map<int, std::vector<void *>> scatter_lists;

    for (auto neighbour : send_neighbours) {
        gather_lists.try_emplace(neighbour);
    }

    for (auto neighbour : recv_neighbours) {
        scatter_lists.try_emplace(neighbour);
    }

    for (auto dat_index : dats) {
        if (partial_exchanges.find(dat_index) != partial_exchanges.end()) {
            continue;
        }

        auto dat = get_dat(dat_index);

        extract_indirect_lists(OP_export_exec_list[dat->set->index], dat, gather_lists);
        extract_indirect_lists(OP_export_nonexec_list[dat->set->index], dat, gather_lists);

        auto import_exec_offset = dat->set->size;
        auto import_nonexec_offset = dat->set->size + OP_import_exec_list[dat->set->index]->size;

        extract_blocked_lists(OP_import_exec_list[dat->set->index], dat,
                              import_exec_offset, scatter_lists);
        extract_blocked_lists(OP_import_nonexec_list[dat->set->index], dat,
                              import_nonexec_offset, scatter_lists);
    }

    for (auto [dat_index, map_index] : partial_exchanges) {
        auto dat = get_dat(dat_index);

        extract_indirect_lists(OP_export_nonexec_permap[map_index], dat, gather_lists);
        extract_indirect_lists(OP_import_nonexec_permap[map_index], dat, scatter_lists);
    }

    auto offset = 0;
    for (auto neighbour : send_neighbours) {
        auto& list = gather_lists[neighbour];
        exchange_info.send_offsets.push_back(offset);

        auto size = 0;
        for (auto ptr : list) {
            exchange_info.ptrs.gather_ptrs.push_back(ptr);
            ++offset;
            ++size;
        }

        exchange_info.send_sizes.push_back(size);
    }

    offset = 0;
    for (auto neighbour : recv_neighbours) {
        auto& list = scatter_lists[neighbour];
        exchange_info.recv_offsets.push_back(offset);

        auto size = 0;
        for (auto ptr : list) {
            exchange_info.ptrs.scatter_ptrs.push_back(ptr);
            ++offset;
            ++size;
        }

        exchange_info.recv_sizes.push_back(size);
    }

    exchange_info.ptrs.gather_ptrs_d =
        (void **) copy_to_device(exchange_info.ptrs.gather_ptrs.data(),
                                 exchange_info.ptrs.gather_ptrs.size() * 8);

    exchange_info.ptrs.scatter_ptrs_d =
        (void **) copy_to_device(exchange_info.ptrs.scatter_ptrs.data(),
                                 exchange_info.ptrs.scatter_ptrs.size() * 8);
}

static ExchangeInfo *get_exchange_info(const std::set<int> &dats,
                                       const std::unordered_map<int, std::set<int>> &maps) {
    if (dats.size() == 0) {
        return nullptr;
    }

    std::map<int, int> partial_exchanges;
    uint64_t exchange_hash = hash_seed_default;

    for (auto index : dats) {
        exchange_hash = hash(index, exchange_hash);

        auto map_elem = maps.find(index);
        if (map_elem == maps.end()) {
            continue;
        }

        auto map_set = map_elem->second;
        if (map_set.size() == 1 && OP_map_partial_exchange[*map_set.begin()]) {
            partial_exchanges[index] = *map_set.begin();
        }
    }

    exchange_hash = hash(UINT64_MAX, exchange_hash);
    for (auto [dat_index, map_index] : partial_exchanges) {
        exchange_hash = hash(map_index, hash(dat_index, exchange_hash));
    }

    auto existing_exchange = exchanges.find(exchange_hash);
    if (existing_exchange != exchanges.end()) {
        return &existing_exchange->second;
    }

    auto [new_exchange, inserted] = exchanges.emplace(exchange_hash, ExchangeInfo());
    assert(inserted);

    init_exchange_info(dats, partial_exchanges, new_exchange->second);
    return &new_exchange->second;
}

int op_mpi_halo_exchanges_unified(op_set set, int nargs, op_arg *args) {
    bool exec = false;
    int size = set->size;

    for (int n = 0; n < nargs; ++n) {
        if (!args[n].opt) continue;
        if (args[n].argtype != OP_ARG_DAT || args[n].idx == -1) continue;
        if (args[n].acc == OP_READ) continue;

        exec = true;
        size += set->exec_size;
        break;
    }

    // Collect list of 4 and 8 byte dats we need to perform exchange for
    std::set<int> dats_4;
    std::set<int> dats_8;

    std::unordered_map<int, std::set<int>> maps;

    for (int n = 0; n < nargs; ++n) {
        if (!args[n].opt) continue;
        if (args[n].argtype != OP_ARG_DAT) continue;
        if (args[n].acc != OP_READ && args[n].acc != OP_RW) continue;
        if (args[n].dat->dirtybit != 1) continue;
        if (!exec && args[n].idx == -1) continue;

        auto elem_size = args[n].dat->size / args[n].dat->dim;
        assert(elem_size == 4 || elem_size == 8);

        if (elem_size == 4) {
            dats_4.insert(args[n].dat->index);
        } else if (elem_size == 8) {
            dats_8.insert(args[n].dat->index);
        }

        if (args[n].map != OP_ID) {
            maps[args[n].dat->index].insert(args[n].map->index);
        }
    }

    wait_scatters();

    active_exchange_4 = get_exchange_info(dats_4, maps);
    active_exchange_8 = get_exchange_info(dats_8, maps);

    if (active_exchange_4 != nullptr || active_exchange_8 != nullptr) {
        auto *ptrs_4 = active_exchange_4 != nullptr ? &active_exchange_4->ptrs : nullptr;
        auto *ptrs_8 = active_exchange_8 != nullptr ? &active_exchange_8->ptrs : nullptr;

        set_ptr_buffers(ptrs_4, ptrs_8);

        realloc_exchange_buffers();
        initiate_gathers();
    }

    return size;
}

template<typename T>
static std::pair<std::vector<MPI_Request>, std::vector<MPI_Request>>
initiate_exchange(const T *gather_buffer, T *scatter_buffer, const ExchangeInfo *exchange) {
    std::vector<MPI_Request> send_reqs;
    std::vector<MPI_Request> recv_reqs;

    if (exchange == nullptr) {
        return std::make_pair(send_reqs, recv_reqs);
    }

    send_reqs.resize(exchange->send_neighbours.size());
    recv_reqs.resize(exchange->recv_neighbours.size());

    assert(sizeof(T) == 4 || sizeof(T) == 8);
    MPI_Datatype exchange_type = sizeof(T) == 4 ? MPI_UINT32_T : MPI_UINT64_T;

    for (int i = 0; i < exchange->send_neighbours.size(); ++i) {
        MPI_Isend(
                (void *) &gather_buffer[exchange->send_offsets[i]],
                exchange->send_sizes[i],
                exchange_type,
                exchange->send_neighbours[i],
                200,
                OP_MPI_WORLD,
                &send_reqs[i]
        );
    }

    for (int i = 0; i < exchange->recv_neighbours.size(); ++i) {
        MPI_Irecv(
                (void *) &scatter_buffer[exchange->recv_offsets[i]],
                exchange->recv_sizes[i],
                exchange_type,
                exchange->recv_neighbours[i],
                200,
                OP_MPI_WORLD,
                &recv_reqs[i]
        );
    }

    return std::make_pair(send_reqs, recv_reqs);
}

void op_mpi_wait_all_unified(int nargs, op_arg *args) {
    if (active_exchange_4 == nullptr && active_exchange_8 == nullptr) {
        return;
    }

    wait_gathers();

    auto [gather_buffer_4, gather_buffer_8] = get_gather_buffers();
    auto [scatter_buffer_4, scatter_buffer_8] = get_scatter_buffers();

    auto [send_reqs_4, recv_reqs_4] = initiate_exchange(gather_buffer_4, scatter_buffer_4,
                                                        active_exchange_4);
    auto [send_reqs_8, recv_reqs_8] = initiate_exchange(gather_buffer_8, scatter_buffer_8,
                                                        active_exchange_8);

    /*
    auto num_req = 0;
    MPI_Request requests[2];

    if (active_exchange_4 != nullptr) {
        MPI_Ineighbor_alltoallv(
                (void *) gather_buffer_4,
                active_exchange_4->send_sizes.data(),
                active_exchange_4->send_offsets.data(),
                MPI_UINT32_T,
                (void *) scatter_buffer_4,
                active_exchange_4->recv_sizes.data(),
                active_exchange_4->recv_offsets.data(),
                MPI_UINT32_T,
                active_exchange_4->comm,
                &requests[num_req++]
        );
    }

    std::vector<MPI_Request> requests_8;
    if (active_exchange_8 != nullptr) {
        MPI_Ineighbor_alltoallv(
                (void *) gather_buffer_8,
                active_exchange_8->send_sizes.data(),
                active_exchange_8->send_offsets.data(),
                MPI_UINT64_T,
                (void *) scatter_buffer_8,
                active_exchange_8->recv_sizes.data(),
                active_exchange_8->recv_offsets.data(),
                MPI_UINT64_T,
                active_exchange_8->comm,
                &requests[num_req++]
        );
    }

    MPI_Waitall(num_req, requests, MPI_STATUSES_IGNORE);
    */

    MPI_Waitall(recv_reqs_4.size(), recv_reqs_4.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_reqs_8.size(), recv_reqs_8.data(), MPI_STATUSES_IGNORE);

    initiate_scatters();

    MPI_Waitall(send_reqs_4.size(), send_reqs_4.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(send_reqs_8.size(), send_reqs_8.data(), MPI_STATUSES_IGNORE);

    if (active_exchange_4 != nullptr) {
        for (auto dat_index : active_exchange_4->dats) {
            auto dat = get_dat(dat_index);
            auto partial = active_exchange_4->partial_exchanges.find(dat_index)
                           != active_exchange_4->partial_exchanges.end();

            dat->dirtybit = partial ? 1 : 0;
            dat->dirty_hd = 2;
        }
    }

    if (active_exchange_8 != nullptr) {
        for (auto dat_index : active_exchange_8->dats) {
            auto dat = get_dat(dat_index);
            auto partial = active_exchange_8->partial_exchanges.find(dat_index)
                           != active_exchange_8->partial_exchanges.end();

            dat->dirtybit = partial ? 1 : 0;
            dat->dirty_hd = 2;
        }
    }

    active_exchange_4 = nullptr;
    active_exchange_8 = nullptr;

    set_ptr_buffers(nullptr, nullptr);
}

std::vector<unsigned> partial_flags;
std::vector<size_t> send_sizes;
std::vector<size_t> recv_sizes;
std::vector<int>      send_neigh_list;
std::vector<int>      recv_neigh_list;
std::vector<size_t> send_offsets;
std::vector<size_t> recv_offsets;
std::vector<MPI_Request> send_requests;
std::vector<MPI_Request> recv_requests;
char *send_buffer_host = NULL;
char *send_buffer_device = NULL;
char *recv_buffer_host = NULL;
char *recv_buffer_device = NULL;
int op2_grp_counter = 0;
int op2_grp_tag = 1234;

extern "C" int op_mpi_halo_exchanges_grouped(op_set set, int nargs, op_arg *args, int device) {
  if (!(device == 2 && OP_unified_exchanges)) {
      deviceSync();
  }

  int size = set->size;
  int direct_flag = 1;

  if (OP_diags > 0) {
    int dummy;
    for (int n = 0; n < nargs; n++)
      op_arg_check(set, n, args[n], &dummy, "halo_exchange_grouped cuda");
  }

  for (int n = 0; n < nargs; n++) {
    if (device == 2 && args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 1) { //Running on device, but dirty on host
      op_upload_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
    if (device == 1 && args[n].opt && args[n].argtype == OP_ARG_DAT &&
        args[n].dat->dirty_hd == 2) { //Running on host, but dirty on device
      op_download_dat(args[n].dat);
      args[n].dat->dirty_hd = 0;
    }
  }

  // check if this is a direct loop
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;

/*  printf("Name: %s, %s, set: %s(%d)\n", OP_kernels[OP_kern_curr].name, direct_flag ? "direct": "indirect", set->name, set->index);
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT)
      printf("\t%s(%d), %s, %d\n", args[n].dat->name, args[n].dat->index, args[n].idx == -1 ? "OP_ID":args[n].map->name,args[n].acc);
  }*/
  if (direct_flag == 1)
    return size;

  if (device == 2 && OP_unified_exchanges) {
      return op_mpi_halo_exchanges_unified(set, nargs, args);
  }

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      size = set->size + set->exec_size;
      exec_flag = 1;
    }
  }

  double c1,c2,t1,t2;
  op_timers_core(&c1, &t1);
  partial_flags.resize(nargs);
  // Fire off any partial halo exchanges that apply
  for (int n = 0; n < nargs; n++) {
    partial_flags[n] = 0;
    if (args[n].opt && args[n].argtype == OP_ARG_DAT) {
      if (args[n].map != OP_ID) {
        // Check if dat-map combination was already done or if there is a
        // mismatch (same dat, diff map)
        int found = 0;
        int fallback = 0;
        for (int m = 0; m < nargs; m++) {
          if (m < n && args[n].dat == args[m].dat && args[n].map == args[m].map) {
            partial_flags[n] = partial_flags[m]==1?2:0;
            found = 1;
          } else if (args[n].dat == args[m].dat && args[n].map != args[m].map)
            fallback = 1;
        }
        // If there was a map mismatch with other argument, do full halo
        // exchange
        if (fallback) continue;
        else if (!found) { // Otherwise, if partial halo exchange is enabled for
                           // this map, do it
          if (OP_map_partial_exchange[args[n].map->index]) {
            partial_flags[n] = 1;
            if (device == 1)
              op_exchange_halo_partial(&args[n], exec_flag);
            else if (device == 2)
              op_exchange_halo_partial_cuda(&args[n], exec_flag);
          }
        }
      }
    }
  }
  std::vector<int> sets;
  sets.resize(0);
  send_sizes.resize(0);
  recv_sizes.resize(0);
  send_neigh_list.resize(0);
  recv_neigh_list.resize(0);
  
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirtybit == 1 && (args[n].acc == OP_READ || args[n].acc == OP_RW) && partial_flags[n] == 0) {
      if ( args[n].idx == -1 && exec_flag == 0) continue; 

      //flag, so same dat not checked again
      args[n].dat->dirtybit = 2;
      //list of sets on which we have data accessed, and list of MPI neighbors
      if (std::find(sets.begin(), sets.end(), args[n].dat->set->index)== sets.end()) {
        sets.push_back(args[n].dat->set->index);
        //receive neighbors
        halo_list imp_exec_list = OP_import_exec_list[args[n].dat->set->index];
        for (int i = 0; i < imp_exec_list->ranks_size; i++)
          if (std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_exec_list->ranks[i])== recv_neigh_list.end()) {
            recv_neigh_list.push_back(imp_exec_list->ranks[i]);
            recv_sizes.push_back(0);
          }
        halo_list imp_nonexec_list = OP_import_nonexec_list[args[n].dat->set->index];
        for (int i = 0; i < imp_nonexec_list->ranks_size; i++)
          if (std::find(recv_neigh_list.begin(), recv_neigh_list.end(), imp_nonexec_list->ranks[i])== recv_neigh_list.end()) {
            recv_neigh_list.push_back(imp_nonexec_list->ranks[i]);
            recv_sizes.push_back(0);
          }

        //send neighbors
        halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
        for (int i = 0; i < exp_exec_list->ranks_size; i++)
          if (std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_exec_list->ranks[i])== send_neigh_list.end()) {
            send_neigh_list.push_back(exp_exec_list->ranks[i]);
            send_sizes.push_back(0);
          }
        halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
        for (int i = 0; i < exp_nonexec_list->ranks_size; i++)
          if (std::find(send_neigh_list.begin(), send_neigh_list.end(), exp_nonexec_list->ranks[i])== send_neigh_list.end()) {
            send_neigh_list.push_back(exp_nonexec_list->ranks[i]);
            send_sizes.push_back(0);
          }
      }
    }
  }
  std::sort(recv_neigh_list.begin(), recv_neigh_list.end());
  std::sort(send_neigh_list.begin(), send_neigh_list.end());
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirtybit == 2 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if ( args[n].idx == -1 && exec_flag == 0) continue; 

      //flag, so same dat not checked again
      args[n].dat->dirtybit = 3;

      //Amount of memory required for send/recv per neighbor
      halo_list imp_exec_list = OP_import_exec_list[args[n].dat->set->index];
      for (int i = 0; i < imp_exec_list->ranks_size; i++) {
        int idx = std::distance(recv_neigh_list.begin(), std::lower_bound(recv_neigh_list.begin(), recv_neigh_list.end(), imp_exec_list->ranks[i]));
        recv_sizes[idx] += (size_t)args[n].dat->size * imp_exec_list->sizes[i];
      }
      halo_list imp_nonexec_list = OP_import_nonexec_list[args[n].dat->set->index];
      for (int i = 0; i < imp_nonexec_list->ranks_size; i++) {
        int idx = std::distance(recv_neigh_list.begin(), std::lower_bound(recv_neigh_list.begin(), recv_neigh_list.end(), imp_nonexec_list->ranks[i]));
        recv_sizes[idx] += (size_t)args[n].dat->size * imp_nonexec_list->sizes[i];
      }
      halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
      for (int i = 0; i < exp_exec_list->ranks_size; i++) {
        int idx = std::distance(send_neigh_list.begin(), std::lower_bound(send_neigh_list.begin(), send_neigh_list.end(), exp_exec_list->ranks[i]));
        send_sizes[idx] += (size_t)args[n].dat->size * exp_exec_list->sizes[i];
      }
      halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
      for (int i = 0; i < exp_nonexec_list->ranks_size; i++) {
        int idx = std::distance(send_neigh_list.begin(), std::lower_bound(send_neigh_list.begin(), send_neigh_list.end(), exp_nonexec_list->ranks[i]));
        send_sizes[idx] += (size_t)args[n].dat->size * exp_nonexec_list->sizes[i];
      }      
    }
  }

  //Realloc buffers
  size_t size_send = std::accumulate(send_sizes.begin(), send_sizes.end(), 0u);
  size_t size_recv = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0u);
  op_realloc_comm_buffer(&send_buffer_host, &recv_buffer_host, &send_buffer_device, &recv_buffer_device, device, size_send, size_recv);

  //Calculate offsets
  send_offsets.resize(send_sizes.size());
  recv_offsets.resize(recv_sizes.size());
  std::fill(send_offsets.begin(), send_offsets.end(), 0u);
  std::fill(recv_offsets.begin(), recv_offsets.end(), 0u);
  if (send_sizes.size()>0) std::partial_sum(send_sizes.begin(), send_sizes.begin()+send_sizes.size()-1, send_offsets.begin()+1);
  if (recv_sizes.size()>0) std::partial_sum(recv_sizes.begin(), recv_sizes.begin()+recv_sizes.size()-1, recv_offsets.begin()+1);

  op2_grp_counter = 0;
  //Pack buffers
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirtybit == 3 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if ( args[n].idx == -1 && exec_flag == 0) continue; 
      //flag, so same dat not checked again
      args[n].dat->dirtybit = 4;
      halo_list exp_exec_list = OP_export_exec_list[args[n].dat->set->index];
      halo_list exp_nonexec_list = OP_export_nonexec_list[args[n].dat->set->index];
      if (device==1) gather_data_to_buffer_ptr     (args[n], exp_exec_list, exp_nonexec_list, send_buffer_host, send_neigh_list, send_offsets );
      if (device==2) gather_data_to_buffer_ptr_cuda(args[n], exp_exec_list, exp_nonexec_list, send_buffer_device, send_neigh_list, send_offsets );
    }
  }

  send_requests.resize(send_neigh_list.size());
  recv_requests.resize(recv_neigh_list.size());
  
  //Non-blocking receive
//  int rank;
//  MPI_Comm_rank(OP_MPI_WORLD, &rank);
  size_t curr_offset = 0;
  op2_grp_tag++;
  for (unsigned i = 0; i < recv_neigh_list.size(); i++) {
    char *buf = (device==2 && OP_gpu_direct) ? recv_buffer_device : recv_buffer_host;
    //printf("rank %d recv %d bytes from %d\n", rank, recv_sizes[i], recv_neigh_list[i]);
    MPI_Irecv(buf + curr_offset, recv_sizes[i], MPI_CHAR, recv_neigh_list[i], op2_grp_tag ,OP_MPI_WORLD, &recv_requests[i]);
    curr_offset += recv_sizes[i];
  }

  if (device == 1) {
    size_t curr_offset = 0;
    for (unsigned i = 0; i < send_neigh_list.size(); i++) {
      char *buf = send_buffer_host;
      // int rank;
      // MPI_Comm_rank(OP_MPI_WORLD, &rank);
      // printf("export from %d to %d, number of elements of size %d | sending:\n ",
      //                 rank, send_neigh_list[i],
      //                 send_sizes[i]);
      // double *b = (double*)(buf + curr_offset);
      // for (int el = 0; el <send_sizes[i]/8; el++)
      //   printf("%g ", b[el]);
      // printf("\n");

      MPI_Isend(buf + curr_offset, send_sizes[i], MPI_CHAR, send_neigh_list[i], op2_grp_tag ,OP_MPI_WORLD, &send_requests[i]);
      curr_offset += send_sizes[i];
    }
  } else if (device == 2 && !OP_gpu_direct) {
      op_download_buffer_async(send_buffer_device, send_buffer_host, size_send);
  } else if (device == 2 && OP_gpu_direct) {
      op_gather_record();
  }

  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;

  return size;
}

extern "C"  void op_mpi_wait_all_grouped(int nargs, op_arg *args, int device) {
  // check if this is a direct loop
  int direct_flag = 1;
  for (int n = 0; n < nargs; n++)
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].idx != -1)
      direct_flag = 0;
  if (direct_flag == 1)
    return;

  if (device == 2 && OP_unified_exchanges) {
      op_mpi_wait_all_unified(nargs, args);
      return;
  }

  // not a direct loop ...
  int exec_flag = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].idx != -1 && args[n].acc != OP_READ) {
      exec_flag = 1;
    }
  }
  double c1,c2,t1,t2;
  op_timers_core(&c1, &t1);

  //Sends are only started here when running async on the device
  if (device == 2) {
    size_t curr_offset = 0;
    if(OP_gpu_direct) op_gather_sync();
    else op_download_buffer_sync();
    for (unsigned i = 0; i < send_neigh_list.size(); i++) {
      char *buf = OP_gpu_direct ? send_buffer_device : send_buffer_host;

      // int rank;
      // MPI_Comm_rank(OP_MPI_WORLD, &rank);
      // op_printf("export from %d to %d, number of elements of size %d | sending:\n ",
      //                 rank, send_neigh_list[i],
      //                 send_sizes[i]);

      // if (rank == 0) {
      //     double *b = (double*)(buf + curr_offset);
      //     int cap = 0;
      //     for (int el = 0; el < send_sizes[i] / 8; el++) {
      //       if (cap == 8) break;
      //       printf("%f ", b[el]);
      //       cap++;
      //   }
      //     printf("\n");
      // }

      // op_printf("rank %d send %d bytes to %d\n", rank, send_sizes[i], send_neigh_list[i]);

      MPI_Isend(buf + curr_offset, send_sizes[i], MPI_CHAR, send_neigh_list[i], op2_grp_tag ,OP_MPI_WORLD, &send_requests[i]);
      curr_offset += send_sizes[i];
    }
  }
  for (int n = 0; n < nargs; n++) {
    if (partial_flags[n]==1) {
      if (device == 1) op_wait_all(&args[n]);
      if (device == 2) op_wait_all_cuda(&args[n]);
    }
  }

  if (recv_neigh_list.size() > 0)
    MPI_Waitall(recv_neigh_list.size(), &recv_requests[0], MPI_STATUSES_IGNORE);

  if (device == 2 && !OP_gpu_direct) {
    size_t size_recv = std::accumulate(recv_sizes.begin(), recv_sizes.end(), 0u);
    op_upload_buffer_async(recv_buffer_device, recv_buffer_host, size_recv);
  }
  op2_grp_counter = 0;
  for (int n = 0; n < nargs; n++) {
    if (args[n].opt && args[n].argtype == OP_ARG_DAT && args[n].dat->dirtybit == 4 && (args[n].acc == OP_READ || args[n].acc == OP_RW)) {
      if (args[n].idx == -1 && exec_flag == 0) continue; 
      halo_list imp_exec_list = OP_import_exec_list[args[n].dat->set->index];
      halo_list imp_nonexec_list = OP_import_nonexec_list[args[n].dat->set->index];
      if (device==1) scatter_data_from_buffer_ptr     (args[n], imp_exec_list, imp_nonexec_list, recv_buffer_host, recv_neigh_list, recv_offsets );
      if (device==2) scatter_data_from_buffer_ptr_cuda(args[n], imp_exec_list, imp_nonexec_list, recv_buffer_device, recv_neigh_list, recv_offsets );
      args[n].dat->dirtybit = 0;
      args[n].dat->dirty_hd = device;
    }
  }
  if (op2_grp_counter>0 && device == 2) op_scatter_sync();

  if (send_neigh_list.size() > 0)
    MPI_Waitall(send_neigh_list.size(), &send_requests[0], MPI_STATUSES_IGNORE);

  send_neigh_list.resize(0);
  recv_neigh_list.resize(0);
  op_timers_core(&c2, &t2);
  if (OP_kern_max > 0)
    OP_kernels[OP_kern_curr].mpi_time += t2 - t1;
}

extern "C" void op_mpi_test_all_grouped(int nargs, op_arg *args) {
  if (recv_neigh_list.size()>0) {
    int result;
    MPI_Test(&recv_requests[0],&result,MPI_STATUS_IGNORE);
  }
}

