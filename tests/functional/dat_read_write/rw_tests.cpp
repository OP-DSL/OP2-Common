// Indirect read-write loops, which need colouring: edges update both their
// nodes and one of a few hub nodes, so neighbouring edges and edges sharing
// a hub conflict.  Not intended to be used with OP_NO_REALLOC.

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

#include "../utility.h"

#define NN 12
#define NHUB 7

// --- Utility functions ---
void check(bool cond, int idx, int rank, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d rank: %d\n", msg, idx, rank);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---
void rw_count(int *c0, int *c1, int *ch) {
  *c0 += 1;
  *c1 += 1;
  *ch += 1;
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppReadWriteTests");

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  auto mesh = generate_1D_umesh<float>(NN * NN, comm_size, my_rank);

  std::vector<int> e2h(mesh.nedge);
  for (int e = 0; e < mesh.nedge; ++e)
    e2h[e] = mesh.ei[e] % NHUB;
  std::vector<int> count(mesh.nnode, 0);

  op_set nodes = op_decl_set(mesh.nnode, "nodes");
  op_set edges = op_decl_set(mesh.nedge, "edges");

  op_map m_e2n = op_decl_map(edges, nodes, 2, mesh.e2n.data(), "edge_to_nodes");
  op_map m_e2h = op_decl_map(edges, nodes, 1, e2h.data(), "edge_to_hub");

  op_dat pn_count = op_decl_dat(nodes, 1, "int", count.data(), "pn_count");

  op_partition("", "", NULL, NULL, NULL);

  int node_size_inc_halo = nodes->size + nodes->exec_size + nodes->nonexec_size;

  // --- Indirect int RW through two maps ---
  {
    op_par_loop(rw_count, "rw_count", edges,
      op_arg_dat(pn_count, 0, m_e2n, 1, "int", OP_RW),
      op_arg_dat(pn_count, 1, m_e2n, 1, "int", OP_RW),
      op_arg_dat(pn_count, 0, m_e2h, 1, "int", OP_RW)
    );

    std::vector<int> fetched(node_size_inc_halo, 0);
    op_fetch_data(pn_count, fetched.data());

    std::vector<int> expected(node_size_inc_halo, 0);
    for (int e = 0; e < edges->size + edges->exec_size; ++e) {
      expected[m_e2n->map[2 * e]] += 1;
      expected[m_e2n->map[2 * e + 1]] += 1;
      expected[m_e2h->map[e]] += 1;
    }

    for (int i = 0; i < nodes->size; ++i)
      check(fetched[i] == expected[i], i, my_rank, "rw_count failed");

    printf("rw_count passed [rank %d]\n", my_rank);
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}
