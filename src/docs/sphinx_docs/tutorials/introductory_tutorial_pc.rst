===================
PC: Preconditioners
===================

TODO
Note: The below is mostly as an example of formatting - it duplicates a lot of what's in :doc:`introductory_tutorial_ksp`

This tutorial uses `PETSc KSP tutorial example 23 <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/examples/tutorials/ex23.c.html>`_ (follow link for the full source).

This tutorial solves a simple tridiagonal system. :math:`Ax=b`, where

.. math::

   A = \left[\begin{array}{ccccc}
    2 && -1  &&    &&         &&   &&  \\
   -1 &&  2  && -1 &&         &&   &&  \\
      && -1  &&  2 && -1      &&   &&  \\
      &&     &&    && \ddots  &&   &&  \\
      &&     &&    && -1      && 2 &&  \\
      \end{array}
   \right], \quad b = A \left[\begin{array}{c} 1 \\ 1 \\ \vdots \\ 1 \\ 1\end{array}\right] = \left[\begin{array}{c} 1 \\ 0 \\ \vdots \\ 0 \\ 1\end{array}\right]

Make sure that ``PETSC_DIR`` and ``PETSC_ARCH`` are set in your environment,
corresponding to a working PETSc installation (as in :doc:`introductory_tutorial_hello`)

Try running the program, on one MPI rank, using the following commands

.. code-block:: bash

  cd $PETSC_DIR/$PETSC_ARCH/src/ksp/ksp/examples/tutorials
  make ex23
  $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -np 1 ./ex23

You should see output which describes the properties of the linear solver, e.g.

.. code-block:: bash

  KSP Object: 1 MPI processes
    type: gmres
      restart=30, using Classical (unmodified) Gram-Schmidt Orthogonalization with no iterative refinement
      happy breakdown tolerance 1e-30
    maximum iterations=10000, initial guess is zero
    tolerances:  relative=1e-07, absolute=1e-50, divergence=10000.
    left preconditioning
    using PRECONDITIONED norm type for convergence test
  PC Object: 1 MPI processes
    type: jacobi
    linear system matrix = precond matrix:
    Mat Object: 1 MPI processes
      type: seqaij
      rows=10, cols=10
      total: nonzeros=28, allocated nonzeros=50
      total number of mallocs used during MatSetValues calls =0
        not using I-node routines


Note: you may want to define a convenient shortcut for ``mpiexec``, for example

.. code-block:: bash

    export PMPI=$PETSC_DIR/$PETSC_ARCH/bin/mpiexec

We will do so and use ``$PMPI`` for further examples.

Once we have created a parallel matrix :math:`A` and a vector :math:`x`, we'd like to (approximately)
solve for :math:`x=A^{-1}b`. In PETSc, this is done with the KSP object, which represents a linear solver.
Just as with other PETSc objects, one creates with reference to an MPI communicator

.. literalinclude:: /../../ksp/ksp/tutorials/ex23.c
  :language: C
  :start-at: KSPCreate
  :end-at: KSPCreate
  :lineno-match:

... TODO more on setting operators, etc. ...

Command line options (TODO:link to manual section) are very useful for experimenting with solvers. KSP accepts
many options - for a complete list see the KSP man page. Try running with some basic
diagnostic options, which are interpreted during the call to KSPSetFromOptions().

.. code-block:: bash

  $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -np 1 ./ex23 -ksp_converged_reason -ksp_monitor

You will see output beginning with the following

.. code-block:: bash

    0 KSP Residual norm 7.071067811865e-01
    1 KSP Residual norm 3.162277660168e-01
    2 KSP Residual norm 1.889822365046e-01
    3 KSP Residual norm 1.290994448736e-01
    4 KSP Residual norm 9.534625892456e-02
    5 KSP Residual norm 8.082545620881e-16
  Linear solve converged due to CONVERGED_RTOL iterations 5
  ...

Note that the residual norm shown is the same norm used for the convergence test.
When experimenting with solvers, one is often interested in the true residual norm
:math:`||b-Ax||_2`, which can be displayed with the help of another command line option

.. code-block:: bash

 $PMPI -n 1 ./ex23 -ksp_monitor_true_residual

.. code-block:: bash

  0 KSP preconditioned resid norm 7.071067811865e-01 true resid norm 1.414213562373e+00 ||r(i)||/||b|| 1.000000000000e+00
  1 KSP preconditioned resid norm 3.162277660168e-01 true resid norm 6.324555320337e-01 ||r(i)||/||b|| 4.472135955000e-01
  2 KSP preconditioned resid norm 1.889822365046e-01 true resid norm 3.779644730092e-01 ||r(i)||/||b|| 2.672612419124e-01
  3 KSP preconditioned resid norm 1.290994448736e-01 true resid norm 2.581988897472e-01 ||r(i)||/||b|| 1.825741858351e-01
  4 KSP preconditioned resid norm 9.534625892456e-02 true resid norm 1.906925178491e-01 ||r(i)||/||b|| 1.348399724926e-01
  5 KSP preconditioned resid norm 8.082545620881e-16 true resid norm 1.368774871884e-15 ||r(i)||/||b|| 9.678699938266e-16
  ...

The linear system here will become badly conditioned as the problem size grows. We can observe this behavior with a command line option, taking advantage of the fact that with a method like GMRES (the default Krylov solver in PETSc) we can monitor the extremal singular values of the matrix, thus estimating the condition number.

.. code-block:: bash

   $PMPI -n 1 ./ex23 -options_left -ksp_monitor_singular_value -n 10

.. code-block:: bash

    0 KSP Residual norm 7.071067811865e-01 % max 1.000000000000e+00 min 1.000000000000e+00 max/min 1.000000000000e+00
    1 KSP Residual norm 3.162277660168e-01 % max 1.118033988750e+00 min 1.118033988750e+00 max/min 1.000000000000e+00
    2 KSP Residual norm 1.889822365046e-01 % max 1.543626320888e+00 min 6.059849680171e-01 max/min 2.547301339733e+00
    3 KSP Residual norm 1.290994448736e-01 % max 1.726989238266e+00 min 3.732397710926e-01 max/min 4.627023624011e+00
    4 KSP Residual norm 9.534625892456e-02 % max 1.819585676605e+00 min 2.517568843597e-01 max/min 7.227550822425e+00
    5 KSP Residual norm 8.082545620881e-16 % max 1.841253532831e+00 min 4.050702638550e-02 max/min 4.545516413148e+01
    ...

.. code-block:: bash

  $PMPI -n 1 ./ex23 -options_left -ksp_monitor_singular_value -n 100

.. code-block:: bash

  0 KSP Residual norm 7.071067811865e-01 % max 1.000000000000e+00 min 1.000000000000e+00 max/min 1.000000000000e+00
  1 KSP Residual norm 3.162277660168e-01 % max 1.118033988750e+00 min 1.118033988750e+00 max/min 1.000000000000e+00
  2 KSP Residual norm 1.889822365046e-01 % max 1.543626320888e+00 min 6.059849680171e-01 max/min 2.547301339733e+00
  3 KSP Residual norm 1.290994448736e-01 % max 1.726989238266e+00 min 3.732397710926e-01 max/min 4.627023624011e+00
  4 KSP Residual norm 9.534625892456e-02 % max 1.819585676605e+00 min 2.517568843597e-01 max/min 7.227550822425e+00
  ...
  500 KSP Residual norm 7.642929037953e-08 % max 1.991704329874e+00 min 7.237993730940e-04 max/min 2.751735361913e+03
  501 KSP Residual norm 7.216650362987e-08 % max 1.992019845203e+00 min 6.896064780635e-04 max/min 2.888632732681e+03
  502 KSP Residual norm 6.979838348466e-08 % max 1.992563014160e+00 min 6.728860387218e-04 max/min 2.961219135925e+03
  ...

As the last example demonstrated, the linear solver, GMRES with Jacobi preconditioning is not very useful, even for this simple problem.

... TODO show how to use direct solver ...

... TODO more on choosing preconditioners, in particular noting how things change with a block preconditoner as you change the number of ranks ...
