void comparethem(int *bound, int *bound2, double *center, double *center2, double *pres, double *pres2) {
  if (bound[0] != bound2[0]) printf("Mismatch bound\n");
  if (center[0] != center2[0]) printf("Mismatch center0\n");
  if (center[1] != center2[1]) printf("Mismatch center1\n");
  if (center[2] != center2[2]) printf("Mismatch center2\n");
  if (pres[0] != pres2[0]) printf("Mismatch pres\n");
}