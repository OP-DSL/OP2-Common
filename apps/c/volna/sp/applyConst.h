inline void applyConst(float *in, float *out, const int *variables) {
  if (*variables & 1) {
    out[0] += *in;
  }
  if (*variables & 2) {
    out[1] += *in;
  }
  if (*variables & 4) {
    out[2] += *in;
  }
  if (*variables & 8) {
    out[3] += *in;
  }
}
