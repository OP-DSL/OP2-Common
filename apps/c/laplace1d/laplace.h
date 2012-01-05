void laplace(Real A[2][2], const Real *x) {
  const Real hinv = 1./(x[1] - x[0]);
  A[0][0] = hinv;
  A[0][1] = -hinv;
  A[1][0] = -hinv;
  A[1][1] = hinv;
}
