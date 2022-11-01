=====================================================
An Introduction to the PETSc Finite Element Interface
=====================================================

The purpose of this tutorial is to introduce the PETSc interface to
finite elements using the simplest test case found at the beginning of
any textbook on finite elements: the Poisson equation. This guide is
implemented as `SNES example 64
<../../../src/snes/tutorials/ex64.c.html>`__. In this tutorial we
will:

* introduce and explain the pieces of the PETSc finite element interface
* build a simple application to use as a test
* demonstrate dimension independence and mesh flexibility

Problem Statement
-----------------

If we have a domain :math:`\Omega` whose boundary is :math:`\Gamma`
and partitioned into two non-overlapping pieces where we will enforce
Dirichlet boundary conditions on :math:`\Gamma_D` and Neumann boundary
conditions on :math:`\Gamma_N`, then we can write the strong form as
follows. Find :math:`u:\Omega \rightarrow \mathbb{R}` such that,

 .. math::
    \begin{align*}
    -\nabla\cdot\left(\nabla u\right) &= f\ \text{ in }\Omega,\\
    u &= g\ \text{ on }\Gamma_{D},\\
    \nabla u \cdot \mathbf{n} &= h\ \text{ on }\Gamma_{N}
    \end{align*}

where :math:`f,g,h` are all scalar functions which return real values
and :math:`\mathbf{n}` is the outward facing normal vector. For the
weak form of the equation we will seek a solution :math:`u \in
\mathcal{U}` which satisfies the Dirichlet condition (:math:`u=g\
\text{ on }\Gamma_D`) such that for any test function :math:`\phi \in
\mathcal{V}\ \text{ where }\ (\phi=0\ \text{ on }\Gamma_{D})`,

 .. math::
    R\left(u\right) = -\int f\phi\ d\Omega
    + \int\nabla u\cdot\nabla \phi\ d\Omega
    - \int\nabla u\cdot \mathbf{n}\ \phi\ d\Gamma_N = 0

Notice that we have expressed the weak form as a residual, as if it
were a nonlinear problem. There are two reasons for this. First, it is
unlikely that you are learning about PETSc to solve linear
problems. They are relatively simple and many tools are available
which are easier to use. So while not a typical approach, by learning
this interface you are better prepared to solve your own problems
which are undoubtedly more complex. Second, the additional overhead in
using a nonlinear solver for a linear problem is minimal.

In this implementation we will use the above problem to setup a
so-called patch_ test. Specifically, we will choose an exact solution
that lives in our finite element space (the constant :math:`g=1.234`
or some other easily recognized number), prescribe only Dirichlet
boundary conditions (:math:`\Gamma_N = \varnothing`), and use no body
force (:math:`f=0`). These kinds of tests are common in finite element
software frameworks because, if the linear algebra error is
sufficiently low, will return the precise answer to machine
precision. They are used as a test that all the pieces of a finite
element implementation are working correctly.

The Mesh and Finite Element Space
---------------------------------

Using the PETSc finite element technology will require you to become
familiar with three main objects. First, we need to create what in
PETSc terminology is called a ``DM``. Originally this stood for
*distributed mesh*, but has come to represent an abstraction which
manages not only a parallel discretization, but also its interactions
with the algebraic solvers. We also need to tell PETSc that we are
using a particular type of ``DM``, the ``DMPLEX``, which provides
support for unstructured type grids. Finally we add the canonical
PETSc ``XXXSetFromOptions`` which will fill in a default mesh and
discretization and allow us to control these from the commandline.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at: /* Initialize
   :end-at: "-dm_view"));

Next, we need to create a ``PetscFE`` which manages a finite element
space.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at: /* Initialize the finite
   :end-at:  (PetscObject) fe));

The PETSc implementation of finite element spaces is more abstract
than what you might encounter in traditional finite element packages,
where each type of element shape and/or polynomial order is a separate
class or object. The ``PetscFE`` defines a finite element space by
specifying a prime space of polynomials and also a dual as presented
in :cite:`Kirby04` and shown below in the output from
``-petscfe_view``. The prime space is usually chosen using polynomial
definitions with the span in mind. PETSc has already implemented these
spaces where the polynomial order can be controlled by
``-petscspace_degree``. The dual space is related to how this prime
space is transformed to provide nodal degrees of freedom which can be
assembled to create a global space with the required level of
continuity across elements. Unless you know what you are doing, you
should not need to change the dual space as it will create elements
with Lagrange-type degrees of freedom no matter what prime space is
chosen. We are also allowing PETSc to decide on what quadrature to use
based on the order of polynomials in our prime space and and then
associating this ``PetscFE`` with our ``DM`` as field 0.

.. code-block:: console

  $ ./ex64 -petscspace_degree 1 -petscfe_view
  PetscFE Object: 1 MPI process
    type: basic
    Basic Finite Element in 2 dimensions with 1 components
    PetscSpace Object: 1 MPI process
      type: poly
      Space in 2 variables with 1 components, size 3
      Polynomial space of degree 1
    PetscDualSpace Object: PetscDualSpace_0x84000001_5 1 MPI process
      type: lagrange
      Dual space with 1 components, size 3
      Continuous Lagrange dual space

The Discrete System
-------------------

The third abstract object we need to use in this example is the
*discrete system* known as a ``PetscDS``. We ask the ``DM`` to create
one and then access it to set a few function pointers to inform PETSc
of the problem physics.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:   /* Setup the discrete system
   :end-at:  g3));

PETSc's connection to the weak form of the problem is at the level of
evaluating the integrand at a point. You are responsible for writing
these functions and connecting them to the appropriate parts of the
``PetscDS``. In the above call, we have associated the functions
``f0`` and ``f1`` with the residual and the function ``g3`` with the
Jacobian.

The documentation for ``PetscDSSetResidual`` informs us that PETSc
uses a first order model, which we write below along with our problem
residual. Note the Neumann term is 0 in this implementation and will
be omitted.

.. math::
   \begin{align*}
   \int_\Omega \phi f_0(u, u_t, \nabla u, x, t) + \nabla\phi \cdot {\vec f}_1(u, u_t, \nabla u, x, t)\ d\Omega&\\
   \text{(PetscDSSetResidual)}&\\\ &\\
   -\int f\phi\ d\Omega + \int\nabla u\cdot\nabla \phi\ d\Omega&\\
   \text{(Our Problem Residual)}&
   \end{align*}

From a visual comparison of the form of the residual with our problem
formulation, we observe that we need to implement 2 functions for our
residual.

.. math::
   \begin{align*}
   f_0(u, u_t, \nabla u, x, t) &= -f\\
   {\vec f}_1(u, u_t, \nabla u, x, t) &= \nabla u
   \end{align*}

The arguments of the model residual reveal that :math:`f_0` and
:math:`{\vec f}_1` can be a function of our unknown :math:`u` or its
time and space and their gradients. However, the call signature of the
implemented function reveals that they can be functions of a great
deal more. Consider the implementation of :math:`f_0` shown below.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:   static void f0
   :end-at:  }

Here we observe that the spatial dimension and number of fields are
passed to this function. We could be solving a coupled problem and
this allows us to compute terms in our weak form which couple
equations together. As our fields could also be vector-valued
functions such as displacement or velocity, the offsets are also
specified so that we can index appropriate parts of the unknown
vector. In addition to an unknown field, we can specify auxilary
fields which are not part of the set of unknowns but also discretized
with a finite element space and have spatial and temporal
gradients. Finally, we could have associated a number of constants as
well.

This function call is detailed to allow for generality, but in our
case, we need to use little of it. In our problem, we could have even
skipped implementing a :math:`f_0` function (because we have assumed
no body force, :math:`f=0`) and simply set its pointer to ``NULL`` in
the call to ``PetscDSSetResidual``, but we will keep it for the sake
of the exposition and to make altering the example easier for the
user. Similarly, we implement a :math:`{\vec f}_1` function.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:   static void f1
   :end-at:  }

We also implement the Jacobian of this residual by setting the
function pointers in the call to ``PetscDSSetJacobian``. The ``g0``
and ``g1`` functions represent the derivative of ``f0`` with respect
to the unknown :math:`u` and :math:`\nabla u`, respectively. We have no
such contributions in this problem and so we set them to ``NULL``. The
remaining ``g2`` and ``g3`` functions represent the same derivatives
but of ``f1``. Since our ``f1`` function is a function of
:math:`\nabla u` then we need to implement this function. To be
completely general, the ``g3`` function is responsible for populating
a flattened ``dim`` by ``dim`` tensor. Since we have no tensor
material parameter in our formulation, the derivative we need is
simply the identity.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:   static void g3
   :end-at:  }

Boundary Conditions
-------------------

The final bit of finite element setup is dedicated to setting the
boundary conditions. In this patch test, we are setting all boundary
values to a constant value via a function ``dirichlet_bc``.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at: static PetscErrorCode dirichlet_bc
   :end-at: }

Although PETSc marks boundaries on meshes that it creates, you may
load a mesh (with ``-dm_plex_filename``) that has not marked
boundaries. To ensure they are marked, we will create a label and call
``DMPlexMarkBoundaryFaces``. Then to set the ``dirichlet_bc`` function
as the source of Dirichlet boundary values, we call
``DMAddBoundary``. As with the residual and Jacobian functions, the
call to this function has a lot of arguments to support more
complicated boundary conditions. In our case, we flag the type of
boundary condition as ``DM_BC_ESSENTIAL``, specify that it applies to
the whole mesh with ``'all'``, and apply it to the label we created
for all marker values of 1. Note that when we pass in our
``dirichlet_bc`` function, we cast it to a ``(void (*)(void))``
function pointer. This is because the type of function used to
populate the boundary conditions depends on the type of boundary
condition set. Thus, PETSc has us pass in a void function pointer
which we then trust PETSc to use correctly internally.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:  /* Setup boundary conditions
   :end-at:  NULL));

Solving the System
------------------

Now we can proceed with creating the unknown vector, the nonlinear
solver, and solving the system. As with the ``DM``, by calling
``SNESSetFromOptions`` we can control the solver behavior from the
commandline. Please note that in order to have the ``DMPlex`` use the
local functions we set, we must also set a flag telling it to do so
with ``DMPlexSetSNESLocalFEM``.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:  /* Create the nonlinear solver
   :end-at:  NULL, u));

Finally, we implement a check on the solution obtained. The function
that we used to set Dirichlet boundary conditions can also be used to
check the error since it is the exact solution. We have to cast it
into an array of function pointers to pass it to ``DMComputeL2Diff``.

.. literalinclude:: /../src/snes/tutorials/ex64.c
   :start-at:  /* Check the error
   :end-at: &error));

Runtime Options
---------------

Since we are using this as a check of the code and the system is quite
small, we can use a direct solver in PETSc. The following are the
necessary options.

* ``-ksp_type preonly`` Sets the linear solver as a method which
  applies the preconditioner exactly one time. Since we plan to use a
  direct solver, this is all we need.
* ``-pc_type lu`` Sets the preconditioner as the :math:`LU`
  decomposition of the system matrix.
* ``-snes_error_if_not_converged`` Changes the behavior of PETSc's
  nonlinear solver to raise an error when not converged.
* ``-petscspace_degree 1`` Sets the polynomial space degree to 1
  (linear elements).

The code that we have written is dimension and mesh topology
independent. If you pass in your own mesh, PETSc will detect the
element types automatically. For the PETSc-created mesh types used in
this example, elements are conceptualized by spatial dimension and a
boolean flag for if simplicial elements are to be used.

* Triangles (default): ``-dm_plex_simplex 1 -dm_plex_dim 2``
* Quadrilaterals: ``-dm_plex_simplex 0 -dm_plex_dim 2``
* Tetrahedra: ``-dm_plex_simplex 1 -dm_plex_dim 3``
* Hexahedra: ``-dm_plex_simplex 0 -dm_plex_dim 3``
* Prisms: ``-dm_plex_cell triangular_prism``

The following are a few extra options that you may find helpful for
exploring how this example works.

* ``-snes_solution_view vtk:out.vtu`` While the plots for this test
  are uninteresting, you may wish to view the solution for other
  applications or to gain confidence that options you are passing are
  achieving the desired effect. This viewer will generate a file that
  you can view with visualization software such as Paraview_.
* ``-dm_refine 1`` The option will apply a uniform refinement to the
  mesh the number of times specified by the option.
* ``-petscfe_view`` Enable to print information about the prime and
  dual spaces used.
* ``-petscspace_view`` Enable to print information about the prime
  space


.. bibliography:: /petsc.bib
   :filter: docname in docnames

.. _patch: https://en.wikipedia.org/wiki/Patch_test_(finite_elements)
.. _Paraview: https://www.paraview.org/
