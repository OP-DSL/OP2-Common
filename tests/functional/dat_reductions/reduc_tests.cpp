// Not intended to be used with OP_NO_REALLOC flag

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"

#include "../utility.h"

#define TOL 1e-9
#define NN 6

// --- Utility functions ---
void check(bool cond, int idx, int rank, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d rank: %d\n", msg, idx, rank);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---
void direct_dat1_inc(const float *dr, float *di) {
  *di += (*dr + 3.25);
}
void direct_dat4_inc(const float *dr, float *di) {
  for (int d = 0; d < 4; ++d)
    di[d] += (dr[d] + 1.325 * d);
}

void indirect_dat1_inc(float *n0i, float *n1i, const float *er) { 
  *n0i += (*er);
  *n1i += (*er);
}
void indirect_dat3_inc(float *n0i, float *n1i, const float *er) { 
  for (int d = 0; d < 3; ++d) {
    n0i[d] += er[d];
    n1i[d] += er[d];
  }
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  auto mesh = generate_1D_umesh<float>(NN * NN, comm_size, my_rank);

  op_set nodes = op_decl_set(mesh.nnode, "nodes");
  op_set edges = op_decl_set(mesh.nedge, "edges");

  op_map m_e2n = op_decl_map(edges, nodes, 2, mesh.e2n.data(), "edge_to_nodes");

  op_dat pe_dat1 = op_decl_dat(edges, 1, "float", mesh.e_dat1.data(), "pe_dat1");
  op_dat pe_dat1_u = op_decl_dat(edges, 1, "float", mesh.e_dat1.data(), "pe_dat1_u");
  op_dat pe_dat4 = op_decl_dat(edges, 4, "float", mesh.e_dat4.data(), "pe_dat4");
  op_dat pe_dat4_u = op_decl_dat(edges, 4, "float", mesh.e_dat4.data(), "pe_dat4_u");

  op_dat pn_dat1 = op_decl_dat(nodes, 1, "float", mesh.n_dat1.data(), "pn_dat1");
  op_dat pn_dat1_u = op_decl_dat(nodes, 1, "float", mesh.n_dat1.data(), "pn_dat1_u");
  op_dat pn_dat3 = op_decl_dat(nodes, 3, "float", mesh.n_dat3.data(), "pn_dat3");
  op_dat pn_dat3_u = op_decl_dat(nodes, 3, "float", mesh.n_dat3.data(), "pn_dat3_u");

  op_partition("PARMETIS", "GEOM", NULL, NULL, NULL);

  int node_size_inc_halo = nodes->size + nodes->exec_size + nodes->nonexec_size;
  int edge_size_inc_halo = edges->size + edges->exec_size + edges->nonexec_size;

  // --- Indirect Dat INC DIM=1 ---
  {
    op_par_loop(indirect_dat1_inc, "indirect_dat1_inc", edges,
      op_arg_dat(pn_dat1_u,  0, m_e2n, 1, "float", OP_INC), 
      op_arg_dat(pn_dat1_u,  1, m_e2n, 1, "float", OP_INC), 
      op_arg_dat(pe_dat1,   -1, OP_ID, 1, "float", OP_READ)
    );

    int node_size_inc_halo = nodes->size + nodes->exec_size + nodes->nonexec_size;

    std::vector<float> fetched(node_size_inc_halo, 0.0);
    op_fetch_data(pn_dat1_u, fetched.data());

    std::vector<float> expected(node_size_inc_halo, 0.0);

    for (int i = 0; i < node_size_inc_halo; ++i) {
      expected[i] = ((float*)pn_dat1->data)[i]; // set original
    }
    
    for (int e = 0; e < edges->size + edges->exec_size; ++e) { // apply the same logic as kernel
      int n0 = m_e2n->map[2 * e];
      int n1 = m_e2n->map[2 * e + 1];

      float edge_val = ((float*)pe_dat1->data)[e];

      expected[n0] += (edge_val);
      expected[n1] += (edge_val);
    }

    for (int i = 0; i < nodes->size; ++i) {
      // printf("*** [%d] indirect_dat1_inc fetched[%d] = %f expected = %f\n", my_rank, i, fetched[i], expected[i]);
      check(std::abs(fetched[i] - expected[i]) < TOL, i, my_rank, "indirect_dat1_inc failed");
    }

    printf("indirect_dat1_inc passed [rank %d]\n", my_rank);
  }

  // --- Indirect Dat INC DIM=3---
  {
    op_par_loop(indirect_dat3_inc, "indirect_dat3_inc", edges,
      op_arg_dat(pn_dat3_u,  0, m_e2n, 3, "float", OP_INC), 
      op_arg_dat(pn_dat3_u,  1, m_e2n, 3, "float", OP_INC), 
      op_arg_dat(pe_dat4,   -1, OP_ID, 4, "float", OP_READ)
    );

    std::vector<float> fetched(node_size_inc_halo * 3, 0.0);
    op_fetch_data(pn_dat3_u, fetched.data());

    std::vector<float> expected(node_size_inc_halo * 3, 0.0);

    for (int i = 0; i < node_size_inc_halo; ++i) {
      for (int d = 0; d < 3; ++d) {
        expected[i * 3 + d] = ((float*)pn_dat3->data)[i * 3 + d]; // set original
      }
    }
    
    for (int e = 0; e < edges->size + edges->exec_size; ++e) { // apply the same logic as kernel
      int n0 = m_e2n->map[2 * e];
      int n1 = m_e2n->map[2 * e + 1];

      float* edge_struct = ((float*)pe_dat4->data) + e * 4;

      for (int d = 0; d < 3; ++d) {
        expected[3 * n0 + d] += (edge_struct[d]); //  * 2.3
        expected[3 * n1 + d] += (edge_struct[d]); //  * 3.1
      }
    }

    for (int i = 0; i < nodes->size; ++i) {
      for (int d = 0; d < 3; ++d) {
        // printf("*** [%d] indirect_dat3_inc fetched[%d] = %f expected = %f\n", my_rank, i*3+d, fetched[i*3+d], expected[i*3+d]);
        check(std::abs(fetched[i * 3 + d] - expected[i * 3 + d]) < TOL, i * 3 + d, my_rank, "indirect_dat3_inc failed");
      }
    }
    
    printf("indirect_dat3_inc passed [rank %d]\n", my_rank);
  }

  // --- Direct Dat INC DIM=1 ---
  {
    op_par_loop(direct_dat1_inc, "direct_dat1_inc", edges, // This could be done as OP_RW
      op_arg_dat(pe_dat1,  -1, OP_ID, 1, "float", OP_READ),
      op_arg_dat(pe_dat1_u, -1, OP_ID, 1, "float", OP_INC)
    );

    std::vector<float> fetched(edge_size_inc_halo, 0.0);
    op_fetch_data(pe_dat1_u, fetched.data());
    
    for (int i = 0; i < edges->size; ++i) {
      const float expected = (2 * ((float*)pe_dat1->data)[i] + 3.25);
      // printf("direct_dat1_inc fetched[%d] = %f expected = %f\n", i, fetched[i], expected);
      check(std::abs(fetched[i] - expected) < TOL, i, my_rank, "read1 failed");
    }

    printf("direct_dat1_inc passed [rank %d]\n", my_rank);
  }

  // --- Direct Dat INC DIM=3 ---
  {
    op_par_loop(direct_dat4_inc, "direct_dat4_inc", edges, // This could be done as OP_RW
      op_arg_dat(pe_dat4,  -1, OP_ID, 4, "float", OP_READ),
      op_arg_dat(pe_dat4_u, -1, OP_ID, 4, "float", OP_INC)
    );

    std::vector<float> fetched(edge_size_inc_halo * 4, 0.0);
    op_fetch_data(pe_dat4_u, fetched.data());
    
    for (int i = 0; i < edges->size; ++i) {
      for (int d = 0; d < 4; ++d) {
        const float expected = 2 * ((float*)pe_dat4->data)[i * 4 + d] + 1.325 * d;
        // printf("direct_dat4_inc fetched[%d] = %f expected = %f\n", i*4+d, fetched[i*4+d], expected);
        check(std::abs(fetched[i * 4 + d] - expected) < TOL, i * 4 + d, my_rank, "read4 failed");
      }
    }

    printf("direct_dat4_inc passed [rank %d]\n", my_rank);
  }

  op_exit();

  return 0;
}