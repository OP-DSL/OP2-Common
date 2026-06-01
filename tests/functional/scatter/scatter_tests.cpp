#include "op_seq.h"
#include "op_profile.h"
#include "../utility.h"

#include <cmath>
#include <vector>

#define TOL 1e-5

void check(bool cond, int idx, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d\n", msg, idx);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---

// Write scalar buf value into dat
void write_scalar(float *buf, float *dat) {
  *dat = *buf * 3.0f;
}

void write_five(const float *buf, float *dat0, float *dat1, const int* idx) {
  dat0[0] = (idx[0] >= 0) ? buf[idx[0]] - 2.0f : 0.0;
  dat0[1] = (idx[1] >= 0) ? buf[idx[1]] - 3.0f : 0.0;
  dat0[2] = (idx[2] >= 0) ? buf[idx[2]] - 4.0f : 0.0;
  dat1[0] = (idx[3] >= 0) ? buf[idx[3]] - 5.0f : 0.0;
  dat1[1] = (idx[4] >= 0) ? buf[idx[4]] - 6.0f : 0.0;
}

// Add buf value into dat (accumulate)
void add_scalar(float *buf, float *dat) {
  *dat += *buf; 
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppScatterTests");

  constexpr int n_nodes = 16;
  constexpr int n_scatter = 12;

  std::vector<float> node_data(n_nodes, 0.0f);
  op_set nodes = op_decl_set(n_nodes, "nodes");
  op_dat dat = op_decl_dat(nodes, 1, "float", node_data.data(), "dat");

  // --- Test 1: basic scatter write (identity-like mapping) ---
  {
    op_buff data_buff = op_create_buff("data_buff", "float", n_scatter, 1);
    op_buff map_to_set  = op_create_buff("map_to_set", "int", n_scatter, 1);

    float  *dbuf = reinterpret_cast<float *>(data_buff->data);
    int    *mbuf = reinterpret_cast<int *>(map_to_set->data);
    for (int i = 0; i < n_scatter; ++i) {
      dbuf[i] = (float)(i + 1) * 2.0f;
      mbuf[i] = i;
    }

    op_par_scatter(data_buff, map_to_set, write_scalar, "write_scalar",
                   op_arg_dat(dat, -1, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fetched(n_nodes, -1.0f);
    op_fetch_data(dat, fetched.data());
    for (int i = 0; i < n_scatter; ++i)
      check(std::fabs(fetched[i] - (float)(i + 1) * 2.0f * 3.0f) < TOL, i, "write_scalar failed");

    op_free_buff(data_buff);
    op_free_buff(map_to_set);
    printf("write_scalar passed\n");
  }

  // --- Test 2: scatter with reversed mapping ---
  {
    op_buff data_buff = op_create_buff("data_buff2", "float", n_scatter, 1);
    op_buff map_to_set  = op_create_buff("map_to_set2", "int", n_scatter, 1);

    float *dbuf = reinterpret_cast<float *>(data_buff->data);
    int   *mbuf = reinterpret_cast<int *>(map_to_set->data);
    for (int i = 0; i < n_scatter; ++i) {
      dbuf[i] = (float)(i + 1);
      mbuf[i] = n_scatter - 1 - i;
    }

    // Zero out dat first
    std::vector<float> zeros(n_nodes, 0.0f);
    op_dat dat2 = op_decl_dat(nodes, 1, "float", zeros.data(), "dat2");

    op_par_scatter(data_buff, map_to_set, write_scalar, "write_scalar_rev",
                   op_arg_dat(dat2, 0, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fetched(n_nodes, -1.0f);
    op_fetch_data(dat2, fetched.data());
    for (int i = 0; i < n_scatter; ++i) {
      int dest = n_scatter - 1 - i;
      check(std::fabs(fetched[dest] - (float)(i + 1) * 3.0f) < TOL, dest, "write_scalar_rev failed");
    }

    op_free_buff(data_buff);
    op_free_buff(map_to_set);
    printf("write_scalar_rev passed\n");
  }

  // --- Test 3: multi-dim map, scatter to two dats ---
  {
    constexpr int n3 = 8;
    op_buff data_buff = op_create_buff("data_buff3", "float", n3, 5);
    op_buff map_to_set  = op_create_buff("map_to_set3", "int", n3, 1);

    float *dbuf = reinterpret_cast<float *>(data_buff->data);
    int   *mbuf = reinterpret_cast<int *>(map_to_set->data);
    for (int i = 0; i < n3; ++i) {
      dbuf[i * 5 + 0] = (float)(i * 10) + 1;
      dbuf[i * 5 + 1] = (float)(i * 3.1) + 2.1;
      dbuf[i * 5 + 2] = (float)(i * 2.87) + 6.1;
      dbuf[i * 5 + 3] = (float)(i * 9.5) + 8.1;
      dbuf[i * 5 + 4] = (float)(i * 1.07) + 3.9;
      mbuf[i] = (n3 - 1 - i);
    }

    std::vector<float> z1(n_nodes * 3, 0.0f);
    std::vector<float> z2(n_nodes * 2, -1.0f);
    op_dat dat4a = op_decl_dat(nodes, 3, "float", z1.data(), "dat4a");
    op_dat dat4b = op_decl_dat(nodes, 2, "float", z2.data(), "dat4b");

    std::vector<int> gbl = {2, 4, 0, -1, 3};
    op_par_scatter(data_buff, map_to_set, write_five, "write_five",
                   op_arg_dat(dat4a, 0, OP_ID, 3, "float", OP_WRITE),
                   op_arg_dat(dat4b, 0, OP_ID, 2, "float", OP_WRITE),
                   op_arg_gbl(gbl.data(), 5, "int", OP_READ));

    std::vector<float> fa(n_nodes * 3, -1.0f), fb(n_nodes * 2, -1.0f);
    op_fetch_data(dat4a, fa.data());
    op_fetch_data(dat4b, fb.data());

    for (int i = 0; i < n3; ++i) {
      int nm = (n3 - 1 - i);

      // printf("{fa0 %f %f %f}\n",
      //    std::fabs(fa[3 * nm + 0] - (float)(i * 2.87 + 6.1 - 2.0f)), fa[3 * nm + 0], (float)(i * 2.87 + 6.1 - 2.0f));
      // printf("{fa1 %f %f %f}\n",
      //    std::fabs(fa[3 * nm + 1] - (float)(i * 1.07 + 3.9 - 3.0f)), fa[3 * nm + 1], (float)(i * 1.07 + 3.9 - 3.0f));
      // printf("{fa2 %f %f %f }\n",
      //    std::fabs(fa[3 * nm + 2] - (float)(i * 10 + 1 - 4.0f)), fa[3 * nm + 2], (float)(i * 10 + 1 - 4.0f));
      // printf("{fb0 %f %f %f}\n",
      //    std::fabs(fb[2 * nm + 0] - (float)(0.0f)), fb[2 * nm + 0], (float)(0.0f));
      // printf("{fb1 %f %f %f}\n",
      //    std::fabs(fb[2 * nm + 1] - (float)(i * 9.5 + 8.1 - 6.0f)), fb[2 * nm + 1], (float)(i * 9.5 + 8.1 - 6.0f));

      check(std::fabs(fa[3 * nm + 0] - (float)(i * 2.87 + 6.1 - 2.0f)) < TOL, i, "write_five dat4a 0 failed");
      check(std::fabs(fa[3 * nm + 1] - (float)(i * 1.07 + 3.9 - 3.0f)) < TOL, i, "write_five dat4a 1 failed");
      check(std::fabs(fa[3 * nm + 2] - (float)(i * 10 + 1 - 4.0f)) < TOL, i, "write_five dat4a 2 failed");
      check(std::fabs(fb[2 * nm + 0] - 0.0f) < TOL, i, "write_five dat4b 0 failed");
      check(std::fabs(fb[2 * nm + 1] - (float)(i * 9.5 + 8.1 - 6.0f)) < TOL, i, "write_five dat4b 1 failed");
    }

    op_free_buff(data_buff);
    op_free_buff(map_to_set);
    printf("write_five passed\n");
  }


  op_profile_end();
  op_profile_output();

  op_exit();
  return 0;
}
