void NumericalFluxes_2(double **maxFacetEigenvalues, double **facetVolumes, double *mesh_CellVolumes, //OP_READ
            double *zeroInit, double *minTimeStep ) //OP_MIN
{
  double local = 0.0;
  for (int j = 0; j < 3; j++) {
    local += *maxFacetEigenvalues[j] * *(facetVolumes[j]);
  }
  zeroInit[0] = 0.0;
  zeroInit[1] = 0.0;
  zeroInit[2] = 0.0;
  zeroInit[3] = 0.0;
  
  *minTimeStep = MIN(*minTimeStep, 2.0 * *mesh_CellVolumes / local);
}
