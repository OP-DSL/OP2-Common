// Constants a loop in another file reads, through a kernel defined in this
// one and a helper defined after its use, both declared on one line.

#include "op_seq.h"

#include "const_files.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

double scale_a = 2.5;
double scale_b = -1.25;

void scale_kernel(double *x) {
  *x = scale_a * *x + shift();
}

int main(int argc, char **argv) {
  op_init(argc, argv, 2);

  constexpr int size = 32;
  op_set set = op_decl_set(size, "set");
  std::vector<double> data(size);
  for (int i = 0; i < size; ++i)
    data[i] = i;
  op_dat dat = op_decl_dat(set, 1, "double", data.data(), "dat");

  op_decl_const(1, "double", &scale_a); op_decl_const(1, "double", &scale_b);

  op_partition("", "", NULL, NULL, NULL);

  scale(set, dat);

  std::vector<double> fetched(size);
  op_fetch_data(dat, fetched.data());
  for (int i = 0; i < size; ++i)
    if (std::abs(fetched[i] - (scale_a * i + scale_b)) > 1e-12) {
      printf("ERROR: element %d is %g, expected %g\n", i, fetched[i], scale_a * i + scale_b);
      op_exit();
      return EXIT_FAILURE;
    }
  printf("const_files passed\n");

  op_exit();
  return 0;
}

double shift() {
  return scale_b;
}
