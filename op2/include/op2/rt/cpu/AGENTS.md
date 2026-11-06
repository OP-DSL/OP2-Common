# cpu: the OpenMP backend

The library is built without OpenMP: each phase runs in the generated code's
OpenMP region (`kernel/cpu`), given a record by its strategy.  Owns its
options and shared plans (`cpu::Backend`), reached through the executor.

- The atomic families are off by default: an atomic increment is a
  compare-and-swap loop on most CPUs.
