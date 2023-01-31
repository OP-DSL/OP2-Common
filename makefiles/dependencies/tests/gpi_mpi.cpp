#include <GASPI.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

	if (MPI_Init (&argc, &argv) != MPI_SUCCESS) {
    return 1;
  }
	
  // Dummy function call to make sure linker doesn't do anything strange -
  // this executable will never actually be run
  gaspi_config_t config;
  gaspi_return_t ret = gaspi_config_get ( &config);

	if (MPI_Finalize () != MPI_SUCCESS) {
    return 2;
  }
	
  if (ret == GASPI_SUCCESS) {
    return 0;
	} else {
    return 3;
	}
}
