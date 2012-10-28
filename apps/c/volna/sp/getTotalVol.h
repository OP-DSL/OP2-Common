inline void getTotalVol(float* cellVolume, float* value, float* totalVol) {
  (*totalVol) += (*cellVolume) * value[0];
}
