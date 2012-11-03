inline void NumericalFluxes(float **maxEdgeEigenvalues, float **EdgeVolumes, float *cellVolumes, //OP_READ
            float *zeroInit, float *minTimeStep ) //OP_MIN
{
  float local = 0.0;
  for (int j = 0; j < 3; j++) {
    local += *maxEdgeEigenvalues[j] * *(EdgeVolumes[j]);
  }
  zeroInit[0] = 0.0;
  zeroInit[1] = 0.0;
  zeroInit[2] = 0.0;
  zeroInit[3] = 0.0;

  *minTimeStep = MIN(*minTimeStep, 2.0 * *cellVolumes / local);
}
