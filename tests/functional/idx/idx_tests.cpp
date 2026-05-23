// Not intended to be used with OP_NO_REALLOC flag

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

#include "../utility.h"

#define TOL 1e-9

// --- Utility functions ---
void check(bool cond, int idx, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d\n", msg, idx);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---
void write_direct_idx(double *dat, const int *idx) { 
  *dat = (double)*idx; 
}

void write_indirect_idx(double *dat, const int *idx0, const int *idx1,
                        const int *idx2) {
  dat[0] = (double)*idx0;
  dat[1] = (double)*idx1;
  dat[2] = (double)*idx2;
}

void write_mixed_idx(double *dat, const int *direct_idx, const int *idx0,
                     const int *idx1, const int *idx2) {
  dat[0] = (double)*direct_idx;
  dat[1] = (double)*idx0;
  dat[2] = (double)*idx1;
  dat[3] = (double)*idx2;
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppIdxTests");

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  constexpr int gbl_edges = 48;
  constexpr int gbl_nodes = 64;
  constexpr int map_dim = 3;

  const int local_edges = compute_local_size(gbl_edges, comm_size, my_rank);
  const int local_nodes = compute_local_size(gbl_nodes, comm_size, my_rank);
  const int edge_start = get_local_start(gbl_edges, comm_size, my_rank);

  std::vector<int> ppedge_data(local_edges * map_dim, 0);
  std::vector<int> qqedge_data(local_edges * map_dim, 0);
  for (int e = 0; e < local_edges; ++e) {
    const int global_edge = edge_start + e;
    for (int d = 0; d < map_dim; ++d) {
      ppedge_data[e * map_dim + d] = (global_edge * 7 + d * 11) % gbl_nodes;
      qqedge_data[e * map_dim + d] = (global_edge * 5 + d * 3 + 1) % gbl_nodes;
    }
  }

  std::vector<double> direct_data(local_edges, 0.0);
  std::vector<double> indirect_data(local_edges * map_dim, 0.0);
  std::vector<double> mixed_data(local_edges * (map_dim + 1), 0.0);

  op_set edges = op_decl_set(local_edges, "edges");
  op_set nodes = op_decl_set(local_nodes, "nodes");
  printf("edge set size = %d, node set size = %d\n", edges->size, nodes->size);

  op_map ppedge = op_decl_map(edges, nodes, map_dim, ppedge_data.data(), "ppedge");
  op_map qqedge = op_decl_map(edges, nodes, map_dim, qqedge_data.data(), "qqedge");
  op_dat direct_dat = op_decl_dat(edges, 1, "double", direct_data.data(), "direct_dat");
  op_dat indirect_dat = op_decl_dat(edges, map_dim, "double", indirect_data.data(), "indirect_dat");
  op_dat mixed_dat = op_decl_dat(edges, map_dim + 1, "double", mixed_data.data(), "mixed_dat");

  op_partition("", "", NULL, NULL, NULL);

  // --- Direct idx: op_arg_idx(-1, OP_ID) ---
  {
    op_par_loop(write_direct_idx, "write_direct_idx", edges,
                op_arg_dat(direct_dat, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_idx(-1, OP_ID));

    std::vector<double> fetched(local_edges, 0.0);
    op_fetch_data(direct_dat, fetched.data());
    
    for (int e = 0; e < local_edges; ++e) {
        const double expected = (double)(e); // it returns the local edge index

        check(std::abs(fetched[e] - expected) < TOL, e,
                "op_arg_idx(-1, OP_ID) failed");
    }

    printf("direct idx passed [rank %d]\n", my_rank);
  }

  // --- Indirect idx: op_arg_idx values from two maps ---
  {
    op_par_loop(write_indirect_idx, "write_indirect_idx", edges,
                op_arg_dat(indirect_dat, -1, OP_ID, map_dim, "double", OP_WRITE),
                op_arg_idx(0, ppedge), 
                op_arg_idx(1, qqedge),
                op_arg_idx(2, qqedge));

    std::vector<double> fetched(local_edges * map_dim, 0.0);
    op_fetch_data(indirect_dat, fetched.data());

    for (int e = 0; e < local_edges; ++e) {
      const double expected0 = (double)ppedge->map[e * map_dim + 0];
      const double expected1 = (double)qqedge->map[e * map_dim + 1];
      const double expected2 = (double)qqedge->map[e * map_dim + 2];

      check(std::abs(fetched[e * map_dim + 0] - expected0) < TOL,
            e * map_dim + 0, "op_arg_idx(0, ppedge) failed");
      check(std::abs(fetched[e * map_dim + 1] - expected1) < TOL,
            e * map_dim + 1, "op_arg_idx(1, qqedge) failed");
      check(std::abs(fetched[e * map_dim + 2] - expected2) < TOL,
            e * map_dim + 2, "op_arg_idx(2, qqedge) failed");
    }

    printf("indirect idx with two maps passed [rank %d]\n", my_rank);
  }

  // --- Combined direct and indirect idx in the same kernel ---
  {
    op_par_loop(write_mixed_idx, "write_mixed_idx", edges,
                op_arg_dat(mixed_dat, -1, OP_ID, map_dim + 1, "double", OP_WRITE),
                op_arg_idx(-1, OP_ID), 
                op_arg_idx(0, ppedge),
                op_arg_idx(1, ppedge), 
                op_arg_idx(2, ppedge));

    std::vector<double> fetched(local_edges * (map_dim + 1), 0.0);
    op_fetch_data(mixed_dat, fetched.data());

    for (int e = 0; e < local_edges; ++e) {
      const int base = e * (map_dim + 1);
      const double expected_direct = (double)(e); // it returns the local edge index
      check(std::abs(fetched[base] - expected_direct) < TOL, base,
              "mixed op_arg_idx direct failed");

      for (int d = 0; d < map_dim; ++d) {
          const double expected = (double)ppedge->map[e * map_dim + d];
          check(std::abs(fetched[base + 1 + d] - expected) < TOL, base + 1 + d,
              "mixed op_arg_idx indirect failed");
      }
    }

    printf("mixed direct and indirect idx passed [rank %d]\n", my_rank);
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}
