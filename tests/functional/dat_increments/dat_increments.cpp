// Functional test for indirect OP_INC handling: each increment form the
// translator accepts, mixed scalar types and dimensions, aliased and optional
// arguments, an increment through a helper function, and repeated loop calls,
// checked element by element against a host reference.  Serial only: the
// reference indexes the global numbering.

#include "op_seq.h"
#include "op_profile.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NN 17
#define NNODE ((NN - 1) * (NN - 1))
#define NOTHER (NNODE + 7)
#define NSWEEPS 20

// --- KERNELS ---
inline void add_scaled(double *target, double scale, double value) {
  target[0] += scale * value;
}

// g aliases a (same dat, map and index) and f aliases d through the reversed
// map.  Each increment form appears once: +=, -=, *p = *p + e, x = x + e,
// x = e + x, x = x - e, ++x, x++, --x, *p +=, a helper, a loop and an if.
inline void resk(const double *weight, const double *u8, const float *u4,
                 double *a, float *b, int *c, double *d, float *e, double *f,
                 double *g, double *r, float *o, const int *oscale,
                 const double *beta) {
  double s = (*beta) * (*weight);
  float sf = (float)s;

  a[0] += s * (u8[0] + (double)u4[0]);
  a[1] -= s * (u8[0] - (double)u4[1]);
  *b = *b + sf * u4[0];
  c[0] = c[0] + ((int)(s * 100.0) % 7) + 1;
  c[1] = ((int)(s * 10.0) % 3) + 1 + c[1];
  ++c[2];
  *d += s;
  e[0]++;
  --e[1];
  add_scaled(f, s, u8[0]);
  g[0] = g[0] + s * (u8[0] + (double)u4[0]);
  g[1] = g[1] - s * (u8[0] - (double)u4[1]);
  for (int k = 0; k < 4; ++k)
    r[k] += s * (k + 1);
  if (*oscale != 0) {
    o[0] += sf * (u4[0] + u4[1]);
  }
}

// --- host reference ---
static int failures = 0;

static void check(const char *name, int elem, int component, double expected, double actual, double tol) {
  if (std::fabs(actual - expected) <= tol * std::fmax(1.0, std::fabs(expected)))
    return;

  if (failures < 20)
    printf("MISMATCH %s element %d component %d expected %.12g got %.12g\n", name, elem, component, expected,
           actual);
  ++failures;
}

int main(int argc, char **argv) {
  op_init(argc, argv, 2);

  const int i_p[4] = {-1, 1, 0, 0};
  const int j_p[4] = {0, 0, -1, 1};

  // A five-point-stencil edge list.
  std::vector<int> pp, pp_rev, to_other;
  std::vector<double> weight;
  for (int i = 0; i < NN - 1; ++i) {
    for (int j = 0; j < NN - 1; ++j) {
      int n = i + j * (NN - 1);
      pp.push_back(n);
      pp.push_back(n);
      weight.push_back(-1.0);

      for (int p = 0; p < 4; ++p) {
        int i2 = i + i_p[p];
        int j2 = j + j_p[p];
        if (i2 < 0 || i2 == NN - 1 || j2 < 0 || j2 == NN - 1)
          continue;

        pp.push_back(n);
        pp.push_back(i2 + j2 * (NN - 1));
        weight.push_back(0.25);
      }
    }
  }

  int nedge = (int)weight.size();
  if (nedge % 128 == 0) {
    printf("dat_increments source size must not be divisible by 128\n");
    exit(EXIT_FAILURE);
  }

  for (int e = 0; e < nedge; ++e) {
    pp_rev.push_back(pp[2 * e + 1]);
    pp_rev.push_back(pp[2 * e]);
    to_other.push_back((5 * e + pp[2 * e] + 2 * pp[2 * e + 1]) % NOTHER);
  }

  // Component- and element-dependent nonzero initial values.
  std::vector<double> u8(NNODE), a(2 * NNODE), d(NNODE), r(4 * NOTHER);
  std::vector<float> u4(2 * NNODE), b(NNODE), e(2 * NNODE), o(NNODE);
  std::vector<int> c(3 * NNODE);
  for (int n = 0; n < NNODE; ++n) {
    u8[n] = 0.5 * n - 0.125;
    for (int k = 0; k < 2; ++k) {
      u4[2 * n + k] = 0.25f * n + 0.03125f * k;
      a[2 * n + k] = 100.0 + 0.01 * n + k;
      e[2 * n + k] = 800.0f + 0.02f * n + k;
    }
    b[n] = 300.0f + 0.03f * n;
    for (int k = 0; k < 3; ++k)
      c[3 * n + k] = 400 + 10 * k + n;
    d[n] = 700.0 + 0.04 * n;
    o[n] = 1000.0f + 0.05f * n;
  }
  for (int n = 0; n < NOTHER; ++n)
    for (int k = 0; k < 4; ++k)
      r[4 * n + k] = 900.0 + 0.01 * n + 0.1 * k;

  std::vector<double> ref_a = a, ref_d = d, ref_r = r;
  std::vector<float> ref_b = b, ref_e = e, ref_o = o;
  std::vector<int> ref_c = c;

  op_set nodes = op_decl_set(NNODE, "nodes");
  op_set other_nodes = op_decl_set(NOTHER, "other_nodes");
  op_set edges = op_decl_set(nedge, "edges");
  op_map ppedge = op_decl_map(edges, nodes, 2, pp.data(), "ppedge");
  op_map ppedge_rev = op_decl_map(edges, nodes, 2, pp_rev.data(), "ppedge_rev");
  op_map edge_to_other = op_decl_map(edges, other_nodes, 1, to_other.data(), "edge_to_other");

  op_dat p_weight = op_decl_dat(edges, 1, "double", weight.data(), "weight");
  op_dat p_u8 = op_decl_dat(nodes, 1, "double", u8.data(), "u8");
  op_dat p_u4 = op_decl_dat(nodes, 2, "float", u4.data(), "u4");
  op_dat p_a = op_decl_dat(nodes, 2, "double", a.data(), "a");
  op_dat p_b = op_decl_dat(nodes, 1, "float", b.data(), "b");
  op_dat p_c = op_decl_dat(nodes, 3, "int", c.data(), "c");
  op_dat p_d = op_decl_dat(nodes, 1, "double", d.data(), "d");
  op_dat p_e = op_decl_dat(nodes, 2, "float", e.data(), "e");
  op_dat p_r = op_decl_dat(other_nodes, 4, "double", r.data(), "r");
  op_dat p_o = op_decl_dat(nodes, 1, "float", o.data(), "o");

  double beta = 2.0;
  op_profile_start("DAT_INCREMENTS");

  for (int sweep = 0; sweep < NSWEEPS; ++sweep) {
    int enable_optional = sweep % 2 == 0;
    int optional_scale = enable_optional;

    for (int edge = 0; edge < nedge; ++edge) {
      int n1 = pp[2 * edge], n2 = pp[2 * edge + 1], other = to_other[edge];
      double s = beta * weight[edge];
      float sf = (float)s;
      double read_u8 = u8[n2];
      float read_u4_0 = u4[2 * n1], read_u4_1 = u4[2 * n1 + 1];

      // a is incremented through two aliased arguments.
      ref_a[2 * n1] += 2.0 * s * (read_u8 + (double)read_u4_0);
      ref_a[2 * n1 + 1] -= 2.0 * s * (read_u8 - (double)read_u4_1);
      ref_b[n2] += sf * read_u4_0;
      ref_c[3 * n1] += ((int)(s * 100.0) % 7) + 1;
      ref_c[3 * n1 + 1] += ((int)(s * 10.0) % 3) + 1;
      ref_c[3 * n1 + 2] += 1;
      // Both d arguments resolve to n2, through different maps.
      ref_d[n2] += s + s * read_u8;
      ref_e[2 * n1] += 1.0f;
      ref_e[2 * n1 + 1] -= 1.0f;
      for (int k = 0; k < 4; ++k)
        ref_r[4 * other + k] += s * (k + 1);
      if (optional_scale != 0)
        ref_o[n1] += sf * (read_u4_0 + read_u4_1);
    }

    op_par_loop(resk, "resk", edges,
                op_arg_dat(p_weight, -1, OP_ID, 1, "double", OP_READ),
                op_arg_dat(p_u8, 1, ppedge, 1, "double", OP_READ),
                op_arg_dat(p_u4, 0, ppedge, 2, "float", OP_READ),
                op_arg_dat(p_a, 0, ppedge, 2, "double", OP_INC),
                op_arg_dat(p_b, 1, ppedge, 1, "float", OP_INC),
                op_arg_dat(p_c, 0, ppedge, 3, "int", OP_INC),
                op_arg_dat(p_d, 1, ppedge, 1, "double", OP_INC),
                op_arg_dat(p_e, 0, ppedge, 2, "float", OP_INC),
                op_arg_dat(p_d, 0, ppedge_rev, 1, "double", OP_INC),
                op_arg_dat(p_a, 0, ppedge, 2, "double", OP_INC),
                op_arg_dat(p_r, 0, edge_to_other, 4, "double", OP_INC),
                op_opt_arg_dat(enable_optional, p_o, 0, ppedge, 1, "float", OP_INC),
                op_arg_gbl(&optional_scale, 1, "int", OP_READ),
                op_arg_gbl(&beta, 1, "double", OP_READ));
  }

  op_fetch_data(p_a, a.data());
  op_fetch_data(p_b, b.data());
  op_fetch_data(p_c, c.data());
  op_fetch_data(p_d, d.data());
  op_fetch_data(p_e, e.data());
  op_fetch_data(p_r, r.data());
  op_fetch_data(p_o, o.data());

  for (int n = 0; n < NNODE; ++n) {
    for (int k = 0; k < 2; ++k) {
      check("a", n, k, ref_a[2 * n + k], a[2 * n + k], 1e-10);
      check("e", n, k, ref_e[2 * n + k], e[2 * n + k], 5e-5);
    }
    check("b", n, 0, ref_b[n], b[n], 5e-5);
    for (int k = 0; k < 3; ++k)
      check("c", n, k, ref_c[3 * n + k], c[3 * n + k], 0.0);
    check("d", n, 0, ref_d[n], d[n], 1e-10);
    check("o", n, 0, ref_o[n], o[n], 5e-5);
  }
  for (int n = 0; n < NOTHER; ++n)
    for (int k = 0; k < 4; ++k)
      check("r", n, k, ref_r[4 * n + k], r[4 * n + k], 1e-10);

  op_profile_end();

  if (failures == 0) {
    printf("Test PASSED\n");
    op_exit();
    return 0;
  }

  printf("Test FAILED with %d mismatches\n", failures);
  op_exit();
  return EXIT_FAILURE;
}
