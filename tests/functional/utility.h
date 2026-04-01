#pragma once

#include <stdio.h>
#include <vector>
#include <numeric>
#include <limits>
#include <string>

#ifdef USE_MPI
#include <mpi.h>

// --- type mapping ---
template <typename T>
MPI_Datatype get_mpi_type();

template <> MPI_Datatype get_mpi_type<float>() { return MPI_FLOAT; }
template <> MPI_Datatype get_mpi_type<int>() { return MPI_INT; }
template <> MPI_Datatype get_mpi_type<double>() { return MPI_DOUBLE; }
#endif

inline int compute_local_size(int global_size, int mpi_comm_size, int mpi_rank) {
  const int base = global_size / mpi_comm_size;
  const int remainder = global_size % mpi_comm_size;
  return base + ((mpi_rank < remainder) ? 1 : 0);
}

inline int get_local_start(int global_size, int mpi_comm_size, int mpi_rank) {
  const int base = global_size / mpi_comm_size;
  const int remainder = global_size % mpi_comm_size;
  return mpi_rank * base + std::min(mpi_rank, remainder);
}

inline void get_rank_and_size(int &mpi_rank, int &mpi_comm_size) {
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_comm_size);
  printf("MPI rank %d of %d\n", mpi_rank, mpi_comm_size);
#endif
}

template <typename T>
void scatter_array(const std::vector<T>& g_array, std::vector<T>& l_array, int mpi_comm_size, 
                    int g_size, int l_size, int dim) {
#ifdef USE_MPI
  std::vector<int> sendcnts(mpi_comm_size), displs(mpi_comm_size);

  int disp = 0;
  for (int i = 0; i < mpi_comm_size; i++) {
    sendcnts[i] = dim * compute_local_size(g_size, mpi_comm_size, i);
  }

  for (int i = 0; i < mpi_comm_size; i++) {
    displs[i] = disp;
    disp += sendcnts[i];
  }

  MPI_Datatype mpi_type = get_mpi_type<T>();

  MPI_Scatterv(g_array.data(),
               sendcnts.data(),
               displs.data(),
               mpi_type,
               l_array.data(),
               l_size * dim,
               mpi_type,
               0,
               MPI_COMM_WORLD);
#else
    l_array = g_array;
#endif
}

template <typename T>
struct Mesh {
  int g_node = 0, g_nedge = 0;
  int nnode = 0, nedge = 0;
  std::vector<int> pp;
  std::vector<T> A, r, u, du;
};

template <typename T>
Mesh<T> generate_simple_umesh(int nn, int mpi_comm_size, int mpi_rank) {
  Mesh<T> m;
  
  m.g_node = (nn - 1) * (nn - 1);
  m.g_nedge = (nn - 1) * (nn - 1) + 4 * (nn - 1) * (nn - 2);

  std::vector<int> g_pp;
  std::vector<T> g_A, g_r, g_u, g_du;

  printf("Global number of nodes, edges = %d, %d\n", m.g_node, m.g_nedge);

  if (mpi_rank == 0) {
    g_pp.resize(2 * m.g_nedge);
    g_A.resize(m.g_nedge);
    g_r.resize(m.g_node);
    g_u.resize(m.g_node);
    g_du.resize(m.g_node);

    // create matrix and r.h.s., and set coordinates needed for renumbering/ partitioning
    int g_e = 0;
    for (int i = 1; i < nn; ++i) {
      for (int j = 1; j < nn; ++j) {
        int g_n = i - 1 + (j - 1) * (nn - 1);
        g_r[g_n] = 0.0;
        g_u[g_n] = 0.0;
        g_du[g_n] = 0.0;

        g_pp[2 * g_e] = g_n;
        g_pp[2 * g_e + 1] = g_n;
        g_A[g_e] = -1.0;
        g_e++;

        for (int pass = 0; pass < 4; ++pass) {
          int i2 = i;
          int j2 = j;
          if (pass == 0)
            i2 += -1;
          if (pass == 1)
            i2 += 1;
          if (pass == 2)
            j2 += -1;
          if (pass == 3)
            j2 += 1;

          if ((i2 == 0) || (i2 == nn) || (j2 == 0) || (j2 == nn)) {
            g_r[g_n] += 0.25;
          } 
          else {
            g_pp[2 * g_e] = g_n;
            g_pp[2 * g_e + 1] = i2 - 1 + (j2 - 1) * (nn - 1);
            g_A[g_e] = 0.25;
            g_e++;
          }
        }
      }
    }
  }

  m.nnode = compute_local_size(m.g_node, mpi_comm_size, mpi_rank);
  m.nedge = compute_local_size(m.g_nedge, mpi_comm_size, mpi_rank);

  printf("Number of nodes, edges on process %d = %d, %d\n", mpi_rank, m.nnode, m.nedge);

  m.pp.resize(2 * m.nedge);
  m.A.resize(m.nedge);
  m.r.resize(m.nnode);
  m.u.resize(m.nnode);
  m.du.resize(m.nnode);

  scatter_array<int>(g_pp, m.pp, mpi_comm_size, m.g_nedge, m.nedge, 2);
  scatter_array<T>(g_A, m.A, mpi_comm_size, m.g_nedge, m.nedge, 1);
  scatter_array<T>(g_r, m.r, mpi_comm_size, m.g_node, m.nnode, 1);
  scatter_array<T>(g_u, m.u, mpi_comm_size, m.g_node, m.nnode, 1);
  scatter_array<T>(g_du, m.du, mpi_comm_size, m.g_node, m.nnode, 1);

  return m;
}

template <typename T>
struct Mesh1D {
  int g_node = 0, g_nedge = 0;
  int nnode = 0, nedge = 0;
  std::vector<int> e2n, ni, ei;
  std::vector<T> e_dat1, e_dat4, n_dat1, n_dat3;
};

template <typename T>
Mesh1D<T> generate_1D_umesh(int nn, int mpi_comm_size, int mpi_rank) {
  Mesh1D<T> m;

  m.g_node = nn;
  m.g_nedge = nn - 1;

  std::vector<int> g_e2n, g_ni, g_ei;
  std::vector<T> ge_dat1, ge_dat4, gn_dat1, gn_dat3;
  printf("Global number of nodes, edges = %d, %d\n", m.g_node, m.g_nedge);

  if (mpi_rank == 0) {
    g_e2n.resize(2 * m.g_nedge);
    
    g_ei.resize(m.g_nedge);
    ge_dat1.resize(m.g_nedge);
    ge_dat4.resize(m.g_nedge * 4);

    g_ni.resize(m.g_node);
    gn_dat1.resize(m.g_node);
    gn_dat3.resize(m.g_node * 3);

    for (int i = 0; i < m.g_nedge; ++i) {
      g_e2n[2 * i] = i;
      g_e2n[2 * i + 1] = i + 1;
      g_ei[i] = i;
      ge_dat1[i] = i * 7.0;
      for (int d = 0; d < 4; ++d)
        ge_dat4[i * 4 + d] = i * 3.0 + 1000.5 * d;
    }

    for (int i = 0; i < m.g_node; ++i) {
      gn_dat1[i] = i * 13;
      g_ni[i] = i;
      for (int d = 0; d < 3; ++d)
        gn_dat3[i * 3 + d] = i * 13 - d * 0.125;
    }
  }

  m.nnode = compute_local_size(m.g_node, mpi_comm_size, mpi_rank);
  m.nedge = compute_local_size(m.g_nedge, mpi_comm_size, mpi_rank);

  printf("Number of nodes, edges on process %d = %d, %d\n", mpi_rank, m.nnode, m.nedge);

  m.e2n.resize(2 * m.nedge);
  m.ei.resize(m.nedge);
  m.e_dat1.resize(m.nedge);
  m.e_dat4.resize(m.nedge * 4);
  m.n_dat1.resize(m.nnode);
  m.n_dat3.resize(m.nnode * 3);
  m.ni.resize(m.nnode);

  scatter_array<int>(g_e2n, m.e2n, mpi_comm_size, m.g_nedge, m.nedge, 2);
  scatter_array<int>(g_ei, m.ei, mpi_comm_size, m.g_nedge, m.nedge, 1);
  scatter_array<T>(ge_dat1, m.e_dat1, mpi_comm_size, m.g_nedge, m.nedge, 1);
  scatter_array<T>(ge_dat4, m.e_dat4, mpi_comm_size, m.g_nedge, m.nedge, 4);
  scatter_array<T>(gn_dat1, m.n_dat1, mpi_comm_size, m.g_node, m.nnode, 1);
  scatter_array<T>(gn_dat3, m.n_dat3, mpi_comm_size, m.g_node, m.nnode, 3);
  scatter_array<int>(g_ni, m.ni, mpi_comm_size, m.g_node, m.nnode, 1);
  
  return m;
}