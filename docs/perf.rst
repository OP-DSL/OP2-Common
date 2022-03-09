.. _perf:

Performance Tuning
==================

Executing with GPUDirect
------------------------

OP2 supports execution with GPU direct MPI when using the MPI + CUDA builds. To enable this, simply pass ``-gpudirect`` as a command line argument when running the executable.

You may also have to user certain environment variables depending on MPI implementation, so check your cluster's user-guide.

OpenMP and OpenMP+MPI
---------------------
It is recommended that you assign one MPI rank per NUMA region when executing MPI+OpenMP parallel code. Usually for a multi-CPU system a single CPU socket is a single NUMA region. Thus, for a 4 socket system, OP2’s MPI+OpenMP code should be executed with 4 MPI processes with each MPI process having multiple OpenMP threads (typically specified by the ``OMP_NUM_THREAD`` flag). Additionally on some systems using ``numactl`` to bind threads to cores could give performance improvements.


.. CUDA arguments
.. --------------
.. tbc