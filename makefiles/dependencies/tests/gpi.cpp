#include <GASPI.h>

int main() {
  // Dummy function call to make sure linker doesn't do anything strange -
  // this executable will never actually be run
  gaspi_config_t config;
  gaspi_return_t ret = gaspi_config_get ( &config);

  if (ret == GASPI_SUCCESS) {
    return 0;
	} else {
    return 1;
	}
}
