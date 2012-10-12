void NumericalFluxes_2(float **maxFacetEigenvalues, float **facetVolumes, float *mesh_CellVolumes, //OP_READ
            float *zeroInit, float *minTimeStep ) //OP_MIN
{
  float local = 0.0;
  for (int j = 0; j < 3; j++) {
    local += *maxFacetEigenvalues[j] * *(facetVolumes[j]);
  }
  zeroInit[0] = 0.0f;
  zeroInit[1] = 0.0f;
  zeroInit[2] = 0.0f;
  zeroInit[3] = 0.0f;
  
  *minTimeStep = 2.0f * *mesh_CellVolumes / local;
}
