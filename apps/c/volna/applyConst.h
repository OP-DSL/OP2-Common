void applyConst(float *in, float *out, int *variables) {
  if (*variables & 1) {
    out[0] += in[0];
  }
  if (*variables & 2) {
    out[1] += in[1];
  }
  if (*variables & 4) {
    out[2] += in[2];
  }
  if (*variables & 8) {
    out[3] += in[3];
  }
}
