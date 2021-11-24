.. _doc_gpus:

===============
PETSc with GPUs
===============

This chapter gives an overview and examples of how to use GPUs with PETSc.
We present things this way (instead of describing GPU-enabled implementations of
various objects in their corresponding chapters), because of the nature of GPU programming:
performance is critical and a cohesive approach to minimize kernel launches
and data movement motivates discussing the interaction of many components.

This presentation focuses on the general design and workflow of using GPU-ready
data and solvers (including :any:`external ones like Hypre <sec_external_gpu_pcs>`),
with explicit examples to orient the user. Additional
information relevant to specific hardware and backends  are often available on
the man pages (which can be reached by clicking on the inline links in the source
below or from the :any:`index <doc_manual_pages>`). Finally, we have
active :any:`mailing lists <doc_mail>` where one can get support (and search archives).


Design
------

Here, we describe PETSc's design choices around GPUs, including policies on CPU/GPU
data mirroring and synchronization,
automatic transfers, initialization, kernel fusion, and support for multiple backends.
Our 2020 report :cite:`mills2020performanceportable` may also be of interest.

The two most mature backends, at the time of this writing, are the CUDA (See e.g. ``MATAIJCUSPARSE``)
and Kokkos backends. See the :any:`doc_gpu_roadmap`.

While OpenMP is not, strictly speaking, a GPU programming paradigm,
we discuss it here as its usage as a backend
presents similar engineering challenges to the usage of GPUs. Future
OpenMP offload support will further blur this distinction.
See :any:`doc_threads` for more-general
notes on PETSc and threads.

Configuring PETSC for GPUs
--------------------------

See :any:`doc_config_accel`. PETSc, specifically ``PetscSF``
:cite:`ZhangBrownBalayFaibussowitschKnepleyMarinMillsMunsonSmithZampini2021`,
can take advantage of CUDA-aware MPI, if configured with an MPI implementation
that supports it.


Vectors on the GPU
------------------

PETSc has several implementations of the ``Vec`` class backed by device memory.
One can use the ``Vec`` API to work with these vectors in much the same way
as their CPU-only counterparts.


Linear operators (Matrices) on the GPU
--------------------------------------

Similarly, several ``Mat`` implementations are defined, backed by device memory
and operating on device ``Vec`` objects. It can be convenient to set
the ``Mat`` type (see ``MatSetType()``) and then use ``MatCreateVecs()`` to create
compatible ``Vec`` objects.


CPU assembly and transfer
~~~~~~~~~~~~~~~~~~~~~~~~~

One can assembly an :any:`AIJ matrices <sec_matsparse>` based on the existing ``MatSetValues``.
This method requires that the matrix be assembled on the CPU first.
See ``Mat`` tutorial examples `ex5k.kokkos.cxx <../../src/mat/tutorials/ex5k.kokkos.cxx.html>`__
and `ex5cu.cu <../../src/mat/tutorials/ex5cu.cu.html>`__ for examples using Kokkos and CUDA, respectively.
Note, this GPU assembly does not communicate off-processor entries, thus, `MAT_IGNORE_OFF_PROC_ENTRIES` is implied.


Matrix-free operators on the GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


On-device matrix assembly
~~~~~~~~~~~~~~~~~~~~~~~~~

One can assemble a matrix using a method based on COO format that works exclusively
on the device and does not currently work on the CPU.
See ``Mat`` test `ex123.c <../../src/mat/tests/ex123.c.html>`__
for an example of the usage of the COO interface.


GPU-enabled solvers
-------------------

Here, we describe some GPU-ready internal and external solvers and preconditioners, with examples.


.. _sec_external_gpu_pcs:

External GPU-enabled solvers as preconditioners
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PETSc can leverage powerful, external, GPU-accelerated solvers, such as Hypre,
as preconditioners.


.. raw:: html

    <hr>

.. bibliography:: /petsc.bib
   :filter: docname in docnames
