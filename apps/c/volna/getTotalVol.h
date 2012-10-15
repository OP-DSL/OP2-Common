void getTotalVol(double* cellVolume, double* value, double* totalVol) {
  (*totalVol) += (*cellVolume) * value[0];
}

