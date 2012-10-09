void NumericalFluxes_2(float **maxFacetEigenvalues, float **FacetVolumes, float *mesh_CellVolumes, //OP_READ
            float *minTimeStep ) //OP_MIN
{
  float local = 0.0;
  for (int j = 0; j < 3; j++) {
    local += *maxFacetEigenvalues[j] * *FacetVolumes[j];
  }

  *minTimeStep = 2.0 * *mesh_CellVolumes / local;
}