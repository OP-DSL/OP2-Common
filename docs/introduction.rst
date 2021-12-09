Introduction
============

`OP2 <https://github.com/OP-DSL/OP2-Common>`_ is a high-level embedded domain specific language (eDSL) for writing **unstructured mesh** algorithms with automatic parellelisation on multi-core and many-core architectures. The API is embedded in both C/C++ and Fortran.

The current OP2 eDSL supports generating code targeting multi-core CPUs with SIMD vectorisation and OpenMP threading, many-core GPUs with CUDA or OpenMP offloading, and distributed memory cluster variants of these using MPI. There is also experimental support for targeting a wider range of GPUs using SYCL and AMD HIP.

These pages provide detailed documentation on using OP2, including an installation guide, an overview of the C++ API, a walkthrough of the development of an example application, and developer documentation.

Licencing
---------

OP2 is released as an open-source project under the BSD 3-Clause License. See the :gh-blob:`LICENSE` file for more information.

Citing
------

To cite OP2, please reference the following paper:

`G. R. Mudalige, M. B. Giles, I. Reguly, C. Bertolli and P. H. J. Kelly, "OP2: An active library framework for solving unstructured mesh-based applications on multi-core and many-core architectures," 2012 Innovative Parallel Computing (InPar), 2012, pp. 1-12, doi: 10.1109/InPar.2012.6339594. <https://ieeexplore.ieee.org/document/6339594>`_

.. code-block:: TeX

   @INPROCEEDINGS{6339594,
     author={Mudalige, G.R. and Giles, M.B. and Reguly, I. and Bertolli, C. and Kelly, P.H.J},
     booktitle={2012 Innovative Parallel Computing (InPar)},
     title={OP2: An active library framework for solving unstructured mesh-based applications on multi-core and many-core architectures},
     year={2012},
     volume={},
     number={},
     pages={1-12},
     doi={10.1109/InPar.2012.6339594}}

Support
-------

The preferred method of reporting bugs and issues with OPS is to submit an issue via the repository’s `issue tracker <https://github.com/OP-DSL/OP2-Common/issues>`_. Users can also email the authors directly by contacting the `OP-DSL team <https://op-dsl.github.io/about.html>`_.

Funding
-------

Development of the OP-DSL libraries is or has been supported by the Engineering and Physical Sciences Research Council, the Royal Society, the Hungarian Academy of Sciences, the European Commission and Rolls-Royce plc., UK AWE, NAG. We are also grateful for hardware resources during development from the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, ARCHER and ARCHER2 UK National Supercomputing Service, the University of Oxford Advanced Research Computing (ARC) facility and hardware donations/access from Nvidia and Intel.
