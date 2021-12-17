#include <cstdio>
#include <hdf5.h>

int main() {
    unsigned maj, min, rel;
    if (H5get_libversion(&maj, &min, &rel) >= 0) {
        std::printf("%d", H5_HAVE_PARALLEL);
    }
}
