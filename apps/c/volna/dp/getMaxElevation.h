void getMaxElevation(double* values, double* currentMaxElevation) {
  double tmp = values[0]+values[3];
  *currentMaxElevation = *currentMaxElevation > tmp ? *currentMaxElevation : tmp;
}
