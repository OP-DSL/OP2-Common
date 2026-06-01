#include "op_seq.h"
#include "op_profile.h"
#include "../utility.h"

#include <cmath>
#include <vector>

#define TOL 1e-9

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
  *dat = *buf;
}

// Write buf[0] into dat0 and buf[1] into dat1
void write_two(float *buf, float *dat0, float *dat1) {
  *dat0 = buf[0];
  *dat1 = buf[1];
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
  // data_buff[i] = (i+1)*2, map_buff[i] = i, expect dat[i] = (i+1)*2
  {
    op_buff data_buff = op_decl_buff<float>(n_scatter, 1, "float", "data_buff");
    op_buff map_buff  = op_decl_buff<int>(n_scatter, 1, "int", "map_buff");

    float  *dbuf = reinterpret_cast<float *>(data_buff->data);
    int    *mbuf = reinterpret_cast<int *>(map_buff->data);
    for (int i = 0; i < n_scatter; ++i) {
      dbuf[i] = (float)(i + 1) * 2.0f;
      mbuf[i] = i;  // identity map (scatter to node i)
    }

    op_par_scatter(data_buff, map_buff, write_scalar, "write_scalar",
                   op_arg_dat(dat, 0, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fetched(n_nodes, -1.0f);
    op_fetch_data(dat, fetched.data());
    for (int i = 0; i < n_scatter; ++i)
      check(std::fabs(fetched[i] - (float)(i + 1) * 2.0f) < TOL, i, "write_scalar failed");

    op_free_buff(data_buff);
    op_free_buff(map_buff);
    printf("write_scalar passed\n");
  }

  // --- Test 2: scatter with reversed mapping ---
  // data_buff[i] = i+1, map_buff[i] = n_scatter-1-i, expect dat[n_scatter-1-i] = i+1
  {
    op_buff data_buff = op_decl_buff<float>(n_scatter, 1, "float", "data_buff2");
    op_buff map_buff  = op_decl_buff<int>(n_scatter, 1, "int", "map_buff2");

    float *dbuf = reinterpret_cast<float *>(data_buff->data);
    int   *mbuf = reinterpret_cast<int *>(map_buff->data);
    for (int i = 0; i < n_scatter; ++i) {
      dbuf[i] = (float)(i + 1);
      mbuf[i] = n_scatter - 1 - i;
    }

    // Zero out dat first
    std::vector<float> zeros(n_nodes, 0.0f);
    op_dat dat2 = op_decl_dat(nodes, 1, "float", zeros.data(), "dat2");

    op_par_scatter(data_buff, map_buff, write_scalar, "write_scalar_rev",
                   op_arg_dat(dat2, 0, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fetched(n_nodes, -1.0f);
    op_fetch_data(dat2, fetched.data());
    for (int i = 0; i < n_scatter; ++i) {
      int dest = n_scatter - 1 - i;
      check(std::fabs(fetched[dest] - (float)(i + 1)) < TOL, dest, "write_scalar_rev failed");
    }

    op_free_buff(data_buff);
    op_free_buff(map_buff);
    printf("write_scalar_rev passed\n");
  }

  // --- Test 3: multi-dim map, scatter to two dats ---
  // data_buff[i] = {i*10, i*100}, map_buff[i] = {i, n_scatter-1-i}
  // dat3a[i] = i*10, dat3b[n_scatter-1-i] = i*100
  {
    constexpr int n3 = 8;
    op_buff data_buff = op_decl_buff<float>(n3, 2, "float", "data3");
    op_buff map_buff  = op_decl_buff<int>(n3, 2, "int", "map3");

    float *dbuf = reinterpret_cast<float *>(data_buff->data);
    int   *mbuf = reinterpret_cast<int *>(map_buff->data);
    for (int i = 0; i < n3; ++i) {
      dbuf[i * 2 + 0] = (float)(i * 10);
      dbuf[i * 2 + 1] = (float)(i * 100);
      mbuf[i * 2 + 0] = i;
      mbuf[i * 2 + 1] = n3 - 1 - i;
    }

    std::vector<float> z(n_nodes, 0.0f);
    op_dat dat3a = op_decl_dat(nodes, 1, "float", z.data(), "dat3a");
    op_dat dat3b = op_decl_dat(nodes, 1, "float", z.data(), "dat3b");

    op_par_scatter(data_buff, map_buff, write_two, "write_two",
                   op_arg_dat(dat3a, 0, OP_ID, 1, "float", OP_WRITE),
                   op_arg_dat(dat3b, 1, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fa(n_nodes, -1.0f), fb(n_nodes, -1.0f);
    op_fetch_data(dat3a, fa.data());
    op_fetch_data(dat3b, fb.data());

    for (int i = 0; i < n3; ++i) {
      check(std::fabs(fa[i] - (float)(i * 10)) < TOL, i, "write_two dat3a failed");
      check(std::fabs(fb[n3 - 1 - i] - (float)(i * 100)) < TOL, n3 - 1 - i, "write_two dat3b failed");
    }

    op_free_buff(data_buff);
    op_free_buff(map_buff);
    printf("write_two passed\n");
  }

  // --- Test 4: op_buff dirty_hd — modify on host, check scatter picks up change ---
  // Reuse data_buff, set dirty_hd=1, modify, scatter again
  {
    constexpr int n4 = 4;
    op_buff data_buff = op_decl_buff<float>(n4, 1, "float", "data4");
    op_buff map_buff  = op_decl_buff<int>(n4, 1, "int", "map4");

    float *dbuf = reinterpret_cast<float *>(data_buff->data);
    int   *mbuf = reinterpret_cast<int *>(map_buff->data);
    for (int i = 0; i < n4; ++i) {
      dbuf[i] = 1.0f;
      mbuf[i] = i;
    }

    std::vector<float> z(n_nodes, 0.0f);
    op_dat dat4 = op_decl_dat(nodes, 1, "float", z.data(), "dat4");

    op_par_scatter(data_buff, map_buff, write_scalar, "write_scalar4a",
                   op_arg_dat(dat4, 0, OP_ID, 1, "float", OP_WRITE));

    // Modify host buffer and mark dirty
    for (int i = 0; i < n4; ++i)
      dbuf[i] = 99.0f;
    data_buff->dirty_hd = 1;

    op_par_scatter(data_buff, map_buff, write_scalar, "write_scalar4b",
                   op_arg_dat(dat4, 0, OP_ID, 1, "float", OP_WRITE));

    std::vector<float> fetched(n_nodes, -1.0f);
    op_fetch_data(dat4, fetched.data());
    for (int i = 0; i < n4; ++i)
      check(std::fabs(fetched[i] - 99.0f) < TOL, i, "dirty_hd update failed");

    op_free_buff(data_buff);
    op_free_buff(map_buff);
    printf("dirty_hd passed\n");
  }

  op_profile_end();
  op_profile_output();

  op_exit();
  return 0;
}
