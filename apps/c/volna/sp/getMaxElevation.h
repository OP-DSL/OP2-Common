inline void getMaxElevation(float* values, float* currentMaxElevation) {
  float tmp = values[0]+values[3];
  *currentMaxElevation = *currentMaxElevation > tmp ? *currentMaxElevation : tmp;
}
