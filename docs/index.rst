Welcome to OP2 documentation!
=============================

Overview
--------

`OP2 <https://github.com/OP-DSL/OP2-Common>`_ (Oxford Parallel library for Unstructured mesh solvers) is a high-level **embedded domain specific language (eDSL)** for writing **unstructured mesh** algorithms with automatic parallelisation on multi-core and many-core architectures, using a single code base. The API is embedded in both C/C++ and Fortran.

The current OP2 eDSL supports generating code targeting multiple architectures: 

* **Multi-core CPUs** with SIMD vectorisation and OpenMP threading
* **Many-core GPUs** using CUDA, HIP, or OpenMP offloading
* **Distributed-memory clusters** using MPI
* **Experimental support** for SYCL-enabled GPUs

OP2 has been used to accelerate a range of computational fluid dynamics (CFD) applications, including full-scale, industrial-grade simulations.

Using OP2, these simulation codes achieve performance portability across CPU and GPU architectures, including NVIDIA, AMD, and Intel GPUs. OP2 can also leverage just-in-time (JIT) compilation of device code, generating C_CUDA or C_HIP code and delivering significant performance improvements. JIT compilation is supported in both the Fortran and C++ backends. This capability has been demonstrated through efficient execution on large-scale supercomputers such as Frontier, ARCHER2, LUMI, Wilkes, and Cirrus.

Hydra is a full-scale, industrial-grade application developed by Rolls-Royce plc for the simulation of aerospace configurations. It has been re-engineered using OP2 to create a performance-portable application, OP2-Hydra, capable of exploiting modern and emerging hardware.

The video below presents an overview of joint work by the OP2 team and Rolls-Royce plc, demonstrating performance-portable CFD simulations in support of virtual certification of gas turbine engines using the OP2-Hydra application.

.. raw:: html

    <div class="video-container">
       <iframe src="https://www.youtube.com/embed/KxCe7dNHyQQ?start=1" title="OP2 overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
    </div>

Licencing
---------

OP2 is released as an open-source project under the **BSD 3-Clause License**. See the :gh-blob:`LICENSE` file for more information.

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

The preferred method for reporting bugs or issues is through the repository’s `issue tracker <https://github.com/OP-DSL/OP2-Common/issues>`_. You can also contact the `OP-DSL team <https://op-dsl.github.io/about.html>`_ directly.

Funding
-------

Development of the OP2 is or has been part supported by:

* Engineering and Physical Sciences Research Council (EPSRC) grants: 
    * EP/1006079/1, EP/100677X/1 on Multi-layered Abstractions for PDEs.
    * EP/JOl0553/1 Algorithms and Software for Emerging Architectures (ASEArch).
    * EP/S005072/1 Strategic Partnership in Computational Science for Advanced Simulation and Modelling of Engineering Systems - ASiMoV.
    * Aerothemal netZero TEChnologies - AZTEC.
    * Virtual Exascale Calculations Transform Aviation - VECTA.
* Royal Society through their Industry Fellowship Scheme (INF/R1/180012).
* The UK Technology Strategy Board and Rolls-Royce plc. through the  SILOET project.
* The Janos Bolyai Research Scholarship of the Hungarian Academy of Sciences and the European Commission. 

We are also grateful for hardware resources during development from:

* The Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
* `ARCHER <http://www.archer.ac.uk/>`_ and `ARCHER2 <https://www.archer2.ac.uk/>`_ UK National Supercomputing Service.
* The `University of Oxford Advanced Research Computing (ARC) <http://dx.doi.org/10.5281/zenodo.22558>`_ facility.
* Hardware donations/access from Nvidia and Intel.

Contents
--------

These pages provide detailed documentation on using OP2, including an installation guide, an overview of the C++ API, a walkthrough of the development of an example application, and developer documentation.

.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   devapp
   api
   translator
   examples
   perf
   developer_guide
   pubs
