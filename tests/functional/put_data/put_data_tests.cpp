// Not intended to be used with OP_NO_REALLOC flag

// op_mpi_put_data is only provided by the MPI library, so this test is only
// built for the MPI variants. The guard is for the translator, which parses
// this file without the MPI include paths.
#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

#include "../utility.h"

#define TOL 1e-9
#define NN 48

// --- Utility functions ---
void check(bool cond, int idx, int rank, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d rank: %d\n", msg, idx, rank);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---
void copy1(double *out, const double *in) { *out = *in; }

void copy3(double *out, const double *in) {
  for (int d = 0; d < 3; ++d)
    out[d] = in[d];
}

void copy4(double *out, const double *in) {
  for (int d = 0; d < 4; ++d)
    out[d] = in[d];
}

// reads both end nodes of an edge, so it touches halo values
void gather2(double *out, const double *n0, const double *n1) {
  out[0] = *n0;
  out[1] = *n1;
}

void poison(double *out) { *out = -999.0; }

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppPutDataTests");

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  auto mesh = generate_1D_umesh<double>(NN, comm_size, my_rank);

  const int orig_nnode = mesh.nnode;
  const int orig_nedge = mesh.nedge;
  const int node_start = get_local_start(mesh.g_node, comm_size, my_rank);
  const int edge_start = get_local_start(mesh.g_nedge, comm_size, my_rank);

  // Initial values distinct from the data we will put later
  std::vector<double> n_init1(orig_nnode, -1.0);
  std::vector<double> n_init3(orig_nnode * 3, -1.0);
  std::vector<double> e_init4(orig_nedge * 4, -1.0);
  std::vector<double> n_out1(orig_nnode, 0.0);
  std::vector<double> n_out3(orig_nnode * 3, 0.0);
  std::vector<double> e_out4(orig_nedge * 4, 0.0);
  std::vector<double> e_gath2(orig_nedge * 2, 0.0);

  op_set nodes = op_decl_set(orig_nnode, "nodes");
  op_set edges = op_decl_set(orig_nedge, "edges");

  op_map m_e2n = op_decl_map(edges, nodes, 2, mesh.e2n.data(), "edge_to_nodes");

  op_dat pn_dat1 = op_decl_dat(nodes, 1, "double", n_init1.data(), "pn_dat1");
  op_dat pn_dat3 = op_decl_dat(nodes, 3, "double", n_init3.data(), "pn_dat3");
  op_dat pe_dat4 = op_decl_dat(edges, 4, "double", e_init4.data(), "pe_dat4");
  op_dat pn_out1 = op_decl_dat(nodes, 1, "double", n_out1.data(), "pn_out1");
  op_dat pn_out3 = op_decl_dat(nodes, 3, "double", n_out3.data(), "pn_out3");
  op_dat pe_out4 = op_decl_dat(edges, 4, "double", e_out4.data(), "pe_out4");
  op_dat pe_gath2 = op_decl_dat(edges, 2, "double", e_gath2.data(), "pe_gath2");

  // Random partition so elements actually move across ranks.
  // The map is required so the edge set is partitioned with the nodes.
  op_partition("RANDOM", "", nodes, NULL, NULL);

  // Indirect loop before any put, so the halos are exchanged and clean
  // (dirtybit == 0) by the time the puts below happen.
  op_par_loop(gather2, "gather2", edges,
              op_arg_dat(pe_gath2, -1, OP_ID, 2, "double", OP_WRITE),
              op_arg_dat(pn_dat1, 0, m_e2n, 1, "double", OP_READ),
              op_arg_dat(pn_dat1, 1, m_e2n, 1, "double", OP_READ));

  std::vector<double> put_n1(orig_nnode);
  std::vector<double> put_n3(orig_nnode * 3);
  std::vector<double> put_e4(orig_nedge * 4);

  for (int i = 0; i < orig_nnode; ++i) {
    const int g = node_start + i;
    put_n1[i] = (double)(g + 1) * 17.0;
    for (int d = 0; d < 3; ++d)
      put_n3[i * 3 + d] = (double)(g + 1) * 17.0 + 0.125 * d;
  }
  for (int i = 0; i < orig_nedge; ++i) {
    const int g = edge_start + i;
    for (int d = 0; d < 4; ++d)
      put_e4[i * 4 + d] = (double)(g + 1) * 3.0 + 1000.5 * d;
  }

  op_mpi_put_data(pn_dat1, put_n1.data(), (size_t)orig_nnode);
  op_mpi_put_data(pn_dat3, put_n3.data(), (size_t)orig_nnode);
  op_mpi_put_data(pe_dat4, put_e4.data(), (size_t)orig_nedge);

  // --- Fetch back into original block order ---
  {
    std::vector<double> fetched(orig_nnode, 0.0);
    op_fetch_data(pn_dat1, fetched.data());
    for (int i = 0; i < orig_nnode; ++i)
      check(std::abs(fetched[i] - put_n1[i]) < TOL, i, my_rank,
            "put/fetch dim=1 nodes failed");
    printf("put/fetch dim=1 nodes passed [rank %d]\n", my_rank);
  }
  {
    std::vector<double> fetched(orig_nnode * 3, 0.0);
    op_fetch_data(pn_dat3, fetched.data());
    for (int i = 0; i < orig_nnode * 3; ++i)
      check(std::abs(fetched[i] - put_n3[i]) < TOL, i, my_rank,
            "put/fetch dim=3 nodes failed");
    printf("put/fetch dim=3 nodes passed [rank %d]\n", my_rank);
  }
  {
    std::vector<double> fetched(orig_nedge * 4, 0.0);
    op_fetch_data(pe_dat4, fetched.data());
    for (int i = 0; i < orig_nedge * 4; ++i)
      check(std::abs(fetched[i] - put_e4[i]) < TOL, i, my_rank,
            "put/fetch dim=4 edges failed");
    printf("put/fetch dim=4 edges passed [rank %d]\n", my_rank);
  }

  // --- Kernel reads after put (covers dirtybit / device upload) ---
  {
    op_par_loop(copy1, "copy1", nodes,
                op_arg_dat(pn_out1, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_dat(pn_dat1, -1, OP_ID, 1, "double", OP_READ));

    std::vector<double> fetched(orig_nnode, 0.0);
    op_fetch_data(pn_out1, fetched.data());
    for (int i = 0; i < orig_nnode; ++i)
      check(std::abs(fetched[i] - put_n1[i]) < TOL, i, my_rank,
            "kernel after put dim=1 nodes failed");
    printf("kernel after put dim=1 nodes passed [rank %d]\n", my_rank);
  }
  {
    op_par_loop(copy3, "copy3", nodes,
                op_arg_dat(pn_out3, -1, OP_ID, 3, "double", OP_WRITE),
                op_arg_dat(pn_dat3, -1, OP_ID, 3, "double", OP_READ));

    std::vector<double> fetched(orig_nnode * 3, 0.0);
    op_fetch_data(pn_out3, fetched.data());
    for (int i = 0; i < orig_nnode * 3; ++i)
      check(std::abs(fetched[i] - put_n3[i]) < TOL, i, my_rank,
            "kernel after put dim=3 nodes failed");
    printf("kernel after put dim=3 nodes passed [rank %d]\n", my_rank);
  }
  {
    op_par_loop(copy4, "copy4", edges,
                op_arg_dat(pe_out4, -1, OP_ID, 4, "double", OP_WRITE),
                op_arg_dat(pe_dat4, -1, OP_ID, 4, "double", OP_READ));

    std::vector<double> fetched(orig_nedge * 4, 0.0);
    op_fetch_data(pe_out4, fetched.data());
    for (int i = 0; i < orig_nedge * 4; ++i)
      check(std::abs(fetched[i] - put_e4[i]) < TOL, i, my_rank,
            "kernel after put dim=4 edges failed");
    printf("kernel after put dim=4 edges passed [rank %d]\n", my_rank);
  }

  // --- Indirect read after put: halos must carry the put values even though
  // --- they were exchanged and clean before the put
  {
    op_par_loop(gather2, "gather2", edges,
                op_arg_dat(pe_gath2, -1, OP_ID, 2, "double", OP_WRITE),
                op_arg_dat(pn_dat1, 0, m_e2n, 1, "double", OP_READ),
                op_arg_dat(pn_dat1, 1, m_e2n, 1, "double", OP_READ));

    std::vector<double> fetched(orig_nedge * 2, 0.0);
    op_fetch_data(pe_gath2, fetched.data());
    for (int i = 0; i < orig_nedge; ++i) {
      // edge g joins nodes g and g + 1
      const int g = edge_start + i;
      check(std::abs(fetched[i * 2] - (double)(g + 1) * 17.0) < TOL, i, my_rank,
            "indirect read after put failed");
      check(std::abs(fetched[i * 2 + 1] - (double)(g + 2) * 17.0) < TOL, i,
            my_rank, "indirect read after put (halo) failed");
    }
    printf("indirect read after put passed [rank %d]\n", my_rank);
  }

  // --- Put over a dat whose device copy is the newer one ---
  {
    op_par_loop(poison, "poison", nodes,
                op_arg_dat(pn_dat1, -1, OP_ID, 1, "double", OP_WRITE));

    op_mpi_put_data(pn_dat1, put_n1.data(), (size_t)orig_nnode);

    op_par_loop(gather2, "gather2", edges,
                op_arg_dat(pe_gath2, -1, OP_ID, 2, "double", OP_WRITE),
                op_arg_dat(pn_dat1, 0, m_e2n, 1, "double", OP_READ),
                op_arg_dat(pn_dat1, 1, m_e2n, 1, "double", OP_READ));

    std::vector<double> fetched(orig_nedge * 2, 0.0);
    op_fetch_data(pe_gath2, fetched.data());
    for (int i = 0; i < orig_nedge; ++i) {
      const int g = edge_start + i;
      check(std::abs(fetched[i * 2] - (double)(g + 1) * 17.0) < TOL, i, my_rank,
            "put over device-resident data failed");
      check(std::abs(fetched[i * 2 + 1] - (double)(g + 2) * 17.0) < TOL, i,
            my_rank, "put over device-resident data (halo) failed");
    }
    printf("put over device-resident data passed [rank %d]\n", my_rank);
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}
