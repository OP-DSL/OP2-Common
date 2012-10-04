void NumericalFluxes_2(double **maxFacetEigenvalues, double **FacetVolumes, double *mesh_CellVolumes, //OP_READ
            double *minTimeStep ) //OP_MIN
{
  double local = 0.0;
  for (int j = 0; j < 3; j++) {
    local += *maxFacetEigenvalues[j] * *FacetVolumes[j];
  }

  *minTimeStep = 2.0 * *mesh_CellVolumes / local;
}