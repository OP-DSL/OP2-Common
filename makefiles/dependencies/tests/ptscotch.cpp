#include <cstdio>

#include <mpi.h>
#include <ptscotch.h>

int main() {
    // Dummy function calls to make sure linker doesn't do anything strange - 
    // this executable will never actually be run

    SCOTCH_Dgraph *g = SCOTCH_dgraphAlloc();
    SCOTCH_dgraphFree(g);

    return 0;
}
