.. _perf:

Performance Tuning
==================

Executing with GPUDirect
------------------------

OP2 supports execution with GPU direct MPI when using the MPI + CUDA builds. 

To enable this, simply pass ``-gpudirect`` as a command line argument when running the executable.

You may also have to user certain environment variables depending on MPI implementation, so check your cluster's user-guide.

OpenMP and OpenMP+MPI
---------------------
It is recommended that you assign one MPI rank per NUMA region when executing MPI+OpenMP parallel code. 

Usually for a multi-CPU system a single CPU socket is a single NUMA region. Thus, for a 4 socket system, OP2's MPI+OpenMP code should be executed with 4 MPI processes with each MPI process having multiple OpenMP threads (typically specified by the ``OMP_NUM_THREADS`` flag). 

Additionally on some systems using ``numactl`` to bind threads to cores could give performance improvements.

numawrap
--------

The ``scripts/numawrap`` script automates NUMA binding and GPU assignment for MPI + GPU runs. It detects the MPI local rank from common launchers (Open MPI, MVAPICH2, Hydra, MPISPAWN) and then:

- Sets ``CUDA_VISIBLE_DEVICES`` to the local rank, ensuring each MPI rank uses a distinct GPU.
- Calls ``numactl --cpunodebind`` to bind the process to the NUMA node corresponding to the local rank (round-robined across the available NUMA nodes).

Usage: pass it as the process wrapper to your MPI launcher:

.. code-block:: shell

   mpirun -np 4 scripts/numawrap ./my_op2_application

This is equivalent to manually calling ``numactl`` per rank but works portably across the MPI implementations above without per-rank launch scripts.


.. CUDA arguments
.. --------------
.. tbc