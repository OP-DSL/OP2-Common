// An info argument of dimension 2: each component comes from the element that
// won that component of the maximum, the two won at opposite ends of the set,
// so by different threads or blocks.

#include "op_seq.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

void argmax2(const double *v, const int *id, double *m, int *at) {
  for (int d = 0; d < 2; ++d)
    if (v[d] > m[d]) {
      m[d] = v[d];
      at[d] = *id;
    }
}

int main(int argc, char **argv) {
  op_init(argc, argv, 2);

  constexpr int size = 4096;
  constexpr int peak0 = 5;
  constexpr int peak1 = size - 3;
  op_set set = op_decl_set(size, "set");
  std::vector<double> values(size * 2);
  std::vector<int> ids(size);
  for (int i = 0; i < size; ++i) {
    values[i * 2] = -double(i - peak0) * (i - peak0);
    values[i * 2 + 1] = -double(i - peak1) * (i - peak1);
    ids[i] = i;
  }
  op_dat v = op_decl_dat(set, 2, "double", values.data(), "v");
  op_dat id = op_decl_dat(set, 1, "int", ids.data(), "id");

  op_partition("", "", NULL, NULL, NULL);

  double m[2] = {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  int at[2] = {-1, -1};
  op_par_loop(argmax2, "argmax2", set,
              op_arg_dat(v, -1, OP_ID, 2, "double", OP_READ),
              op_arg_dat(id, -1, OP_ID, 1, "int", OP_READ),
              op_arg_gbl(m, 2, "double", OP_MAX),
              op_arg_info(at, 2, "int", 2));

  if (m[0] != 0.0 || m[1] != 0.0 || at[0] != peak0 || at[1] != peak1) {
    printf("ERROR: max {%g, %g} at {%d, %d}, expected {0, 0} at {%d, %d}\n", m[0], m[1], at[0],
           at[1], peak0, peak1);
    op_exit();
    return EXIT_FAILURE;
  }
  printf("info passed\n");

  op_exit();
  return 0;
}
