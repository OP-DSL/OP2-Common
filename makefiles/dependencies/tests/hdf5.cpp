#include <cstdio>
#include <hdf5.h>

int main() {
    unsigned maj, min, rel;
    if (H5get_libversion(&maj, &min, &rel) >= 0) {
#ifdef H5_HAVE_PARALLEL
        std::printf("1");
#else
        std::printf("0");
#endif
    }
}
