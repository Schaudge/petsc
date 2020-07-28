======================
Mat: Matrices in PETSc
======================

.. role:: math(raw)
   :format: html latex
..

This tutorial is an introduction to matrix objects (``Mat``) in PETSc.
Working with a matrix in PETSc involves several steps:

-  Create a matrix

-  Insert or add values into the matrix

-  Process (or assemble) the matrix

-  Use the matrix for computations

-  Destroy the matrix

In this tutorial, we demonstrate how to create a tridiagonal parallel matrix.
A matrix of this form can arise, for example, from a three point finite difference discretization of the one-dimensional Laplace equation.
We will also show basic matrix and matrix-vector operations. 
An example of creating a sequential tridiagonal matrix can be found in `PETSc KSP tutorial ex1 <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex1.c>`_
and a parallel example is shown in `PETSc KSP tutorial ex23 <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex23.c>`_.

Creating a PETSc matrix
=======================

We start by setting up a PETSc program.
To utilize the matrix class, we need to include ``petscmat.h`` which 
automatically includes header files for basic PETSc routines, index sets,
and viewers. 
We also need to declare variables to hold the data we will use in this tutorial.
In this case, we declare a ``Mat`` and two ``Vec`` objects, along with other
``PetscInt`` and ``PetscScalar`` variables that will be used to, for example, set matrix values.
PETSc programs usually begin with ``PetscInitialize``, which sets up MPI, among other things. 

::

	static char help[] = "Create a tridiagonal matrix.\n\n";

	#include <petscmat.h>

	int main(int argc,char **args)
	{

	  Mat            A;         
	  Vec            x,b;
	  PetscInt       i,n=10,col[3],rstart,rend,nlocal,mlocal;
	  PetscErrorCode ierr;
	  PetscScalar    value[3];

	  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

We create a matrix with the ``MatCreate`` function. 
This will create a sequential matrix if the program is executed with one processor 
or a parallel matrix if it is executed with more than one processor. 
PETSc supports a variety of matrix implementations since no single matrix format is appropriate for all problems. 
For example, sparse matrices can be stored in compressed row (AIJ) format.
The matrix type can be coded into a program or set from the command line using, for example, ``-mat_type mpiaij``
in the same way that vector formats can be set.
We create an ``n`` by ``n`` square matrix, let PETSc decide how to partition it across processors,
and check for command line options. 

::

	  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
	  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
	  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
	  ierr = MatSetUp(A);CHKERRQ(ierr);

To insert or add entries to a matrix, we use the ``MatSetValues`` function. 
A processor can set any entry in a matrix even if the entry is not locally owned,
but for efficient assembly it is important that each processor 
contributes to entries it owns locally (as much as is possible). 
To accomplish this, we identify the portion of the matrix on each processor
and insert values into the matrix.
In this example, we insert values one row at a time in a loop by providing the column indices and values. 
We handle the boundaries separately since these rows only contain two entries. 
The final step before we can use the matrix is assembly.

::

	  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);

	  /* Boundary */
	  if (!rstart) {
	    rstart = 1;
	    i      = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
	    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
	  }
	  if (rend == n) {
	    rend = n-1;
	    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
	    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
	  }

	  /* Interior */
	  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
	  for (i=rstart; i<rend; i++) {
	    col[0] = i-1; col[1] = i; col[2] = i+1;
	    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
	  }

	  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

Creating a compatible vector
============================

Matrices are often used with vectors in computations. 
When we create a parallel vector to be used with a matrix, 
we can force it to have the same layout by specifying the local and global size of the vector. 
Note that we can create one vector from scratch and duplicate it as needed.

::

	ierr = MatGetLocalSize(A,&nlocal,&mlocal);CHKERRQ(ierr);

	ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
	ierr = VecSetSizes(x,mlocal,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(x);CHKERRQ(ierr);
	ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
	ierr = VecSet(x,1.0);CHKERRQ(ierr);

Matrix operations in PETSc
==========================

PETSc supports many matrix routines such as those that compute the trace of a matrix and matrix norms. 
There are also routines that operate on matrices like scaling a matrix or computing its conjugate. 
Many matrix-vector operations are supported, as well. 
A full list of matrix routines can be found `here <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/index.html>`_
and a few are shown below. 

::

	  ierr = MatMult(A,x,b);CHKERRQ(ierr);
	  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

	  ierr = MatScale(A,2.0);CHKERRQ(ierr);
	  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

All PETSc objects should be destroyed when they are no longer needed,
so we conclude by freeing work space and finalizing the program.

::

	  ierr = MatDestroy(&A);CHKERRQ(ierr);
	  ierr = VecDestroy(&x);CHKERRQ(ierr); 
	  ierr = VecDestroy(&b);CHKERRQ(ierr); 

	  ierr = PetscFinalize();
	  return ierr;
	}

Running this program with two MPI ranks produces the output below.

::

	Vec Object: 2 MPI processes
	  type: mpi
	Process [0]
	1.
	0.
	0.
	0.
	0.
	Process [1]
	0.
	0.
	0.
	0.
	1.
	Mat Object: 2 MPI processes
	  type: mpiaij
	row 0: (0, 4.)  (1, -2.) 
	row 1: (0, -2.)  (1, 4.)  (2, -2.) 
	row 2: (1, -2.)  (2, 4.)  (3, -2.) 
	row 3: (2, -2.)  (3, 4.)  (4, -2.) 
	row 4: (3, -2.)  (4, 4.)  (5, -2.) 
	row 5: (4, -2.)  (5, 4.)  (6, -2.) 
	row 6: (5, -2.)  (6, 4.)  (7, -2.) 
	row 7: (6, -2.)  (7, 4.)  (8, -2.) 
	row 8: (7, -2.)  (8, 4.)  (9, -2.) 
	row 9: (8, -2.)  (9, 4.) 

Further reading
===============

More information about the ``Mat`` class can be found in the documentation. 

-  `A complete list of matrix routines <https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/index.html>`_

-  `PETSc examples using matrices <https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/index.html>`_
