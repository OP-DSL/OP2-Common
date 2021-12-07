#include <cstdio>
#include <hdf5.h>

int main() {
    std::printf("%d", H5_HAVE_PARALLEL);
}
