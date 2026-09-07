.. _perf:

Performance Tuning
==================

Executing with GPUDirect
------------------------

OP2 supports execution with GPU direct MPI when using the MPI + CUDA builds.

This is detected automatically at ``op_init`` and needs no flag: OP2 asks the MPI
library whether it is CUDA or ROCm aware (``MPIX_Query_cuda_support`` and
``MPIX_Query_rocm_support``), and for the MPI implementations that offer no such
query it falls back to the vendor variables ``MPICH_GPU_SUPPORT_ENABLED``,
``I_MPI_OFFLOAD`` and ``MV2_USE_CUDA``. The decision is printed once at startup as
``OP2: GPU-direct MPI = ...`` along with the mechanism that decided it.

Detection only ever turns the feature *on* when it can confirm support, because
handing a device pointer to an MPI that cannot take one fails inside the transport,
whereas staging through host memory is always correct and merely slower.

Set ``OP2_GPU_DIRECT`` to override the detection in either direction: ``1`` forces
it on for an MPI that supports it but does not advertise it, ``0`` forces the
staged path for comparison or to work around a broken transport.

Note that some MPI implementations need their own environment variables set before
they will accept device pointers at all, so check your cluster's user guide.

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