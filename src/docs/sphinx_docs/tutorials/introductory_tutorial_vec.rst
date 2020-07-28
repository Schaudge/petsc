=====================
Vec: Vectors in PETSc
=====================


.. role:: math(raw)
   :format: html latex
..

Vectors are used to store  solutions and right-hand sides for linear
systems of the form :math:`Ax=b`. 
This tutorial is an introduction to vector objects (``Vec``) in PETSc and
is based on `PETSc Vec tutorial ex1 <https://www.mcs.anl.gov/petsc/petsc-current/src/vec/vec/examples/tutorials/ex1.c.html>`_
and `PETSc Vec tutorial ex3 <https://www.mcs.anl.gov/petsc/petsc-current/src/vec/vec/examples/tutorials/ex3.c.html>`_.
This tutorial demonstrates how to:

-  Create PETSc ``Vec`` objects

-  Perform operations on vectors

Creating PETSc vectors
======================

This program creates two PETSc vectors and performs basic operations on them.
To utilize the vector class, we need to include ``petscvec.h`` which 
automatically includes header files for basic PETSc routines, index sets,
and viewers. 
MPI is set up using ``PetscInitialize`` which invokes ``MPI_Init``
and reads the command line for PETSc runtime options. In this case, we give the 
variable ``n`` a default value of 20 with the option of specifying it at runtime.

::

      static char help[] = "Using vectors in PETSc.\n\n";

      #include <petscvec.h>

      int main(int argc,char **argv)
      {
        PetscErrorCode ierr;
        Vec            x,y;
        PetscInt       i,istart,iend,n = 20;
        PetscReal      norm;
        PetscScalar    v,dot,one = 1.0,two = 2.0;

        ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
        ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

PETSc supports sequential and parallel vectors. The type can be set in the code or
determined at runtime. 
Below, we let the vector type be specified at runtime using command line arguments ``-vec_type seq`` or ``-vec_type mpi``.  
The vector type is then set by ``VecSetFromOptions()``.
Alternately, sequential vectors can be created with ``VecCreateSeq()`` and parallel
with ``VecCreateMPI()``.

We set the size of a vector by passing the global size ``n`` and ``PETSC_DECIDE`` for the
local size to ``VecSetSizes()``. PETSc will choose a reasonable partitioning trying to put nearly an
equal number of elements on each processor.
Alternately, we can pass the local size and ``PETSC_DECIDE`` and PETSc will
compute the global size. We can duplicate a vector to create another vector,
which will have the same format and partitioning as the first.

::

      ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
      ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
      ierr = VecSetFromOptions(x);CHKERRQ(ierr);
      ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

Setting vector values
=====================

We can set all entries in a vector to a single value with ``VecSet()``.
Elements of a vector can be set with the ``VecSetValues()`` function.
Parallel vectors in PETSc are partitioned into continguous chunks
across processors. We determine which chunks of the vector are locally 
owned and set each element using global indexing.
A process can set an element in any location, even elements it does not own.
Contributions to a vector can either be summed (``ADD_VALUES``) or replaced (``INSERT_VALUES``).
We assemble a vector in a two-step process. Computation can be done while 
messages are in transition by placing code between these two statements.

::

    ierr = VecSet(x,one);CHKERRQ(ierr);
    ierr = VecSet(y,two);CHKERRQ(ierr);

    VecGetOwnershipRange(x,&istart,&iend);

    for (i=istart; i<iend; i++) {
      v = (PetscScalar)(i);
      VecSetValues(x,1,&i,&v,ADD_VALUES);
    }
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

Vector operations
=================

PETSc supports many operations on vectors and vector routines such as dot products and
norms. Some of these are demonstrated here.
We can visualize vectors and check our results by printing to the consol using 
``VecView()`` and PETSc print function ``PetscPrintf()``.
For large vectors, a ``PetscViewer`` can be useful to create Matlab or HDF5 compatible files.

::

      ierr = VecScale(x,two);CHKERRQ(ierr);
      ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

      ierr = VecDot(x,y,&dot);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Dot of x and y: %g\n",dot);CHKERRQ(ierr);
      ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of x: %g\n",norm);CHKERRQ(ierr);

Finally, we free memory that was used during our program and
call ``PetscFinalize`` which finalizes PETSc libraries and MPI. 

::

      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecDestroy(&y);CHKERRQ(ierr);

      ierr = PetscFinalize();
      return ierr;
    }

Run a PETSc program
===================

We use the command line to run PETSc programs. Navigate to ``ex1.c``
using the ``cd`` command followed by the path to the file. PETSc makes
use of ``make`` to build tests and tutorials, so compiling ``ex1.c`` is
straightforward and we get an executable called ``ex1``. We run this 
program in parallel with two MPI ranks. Additionaly, we set the vector
size and type with command line options. 

.. code-block:: bash

    $ cd $PETSC_DIR/$PETSC_ARCH/src/vec/vec/examples/tutorials
    $ make ex1
    $ $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -np 2 ./ex1 -n 10 -vec_type mpi

Running this program with these options gives us the following output.

.. code-block:: bash

  Vec Object: 2 MPI processes
    type: mpi
  Process [0]
  2.
  4.
  6.
  8.
  10.
  Process [1]
  12.
  14.
  16.
  18.
  20.
  Dot of x and y: 220.
  Norm of x: 39.2428

Further reading
===============

More information about the ``Vec`` class can be found in the documentation. 

-  `A complete list of vector routines <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/index.html>`_

-  `PETSc examples using vectors <https://www.mcs.anl.gov/petsc/petsc-current/src/vec/vec/examples/tutorials/index.html>`_
