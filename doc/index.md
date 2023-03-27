# PETSc {{ version }}

PETSc, the Portable, Extensible Toolkit for Scientific Computation,
pronounced PET-see ([/ˈpɛt-siː/](https://en.wikipedia.org/wiki/Help:IPA/English#Key)), is
for the scalable (parallel) solution of scientific
applications modeled by partial differential equations. It has bindings for C, Fortran, and Python (via {any}`petsc4py<petsc4py_api>`).
PETSc also contains Tao, the Toolkit for Advanced Optimization, software library.
It supports MPI, and GPUs through
CUDA, HIP or OpenCL, as well as hybrid MPI-GPU parallelism; it also supports the NEC-SX Tsubasa Vector Engine.
Immediately jump in and run PETSc code {any}`handson`.

PETSc is developed as {ref}`open-source <doc_license>`, requests and contributions are welcome.

## News

:::{admonition} News: PETSc 2023 Annual Meeting
Registration now open for {any}`The PETSc 2023 Annual Meeting<2023_meeting>`, June 5-7 on the campus of IIT in Chicago.
:::

:::{admonition} News: New Book on PETSc
**PETSc for Partial Differential Equations: Numerical Solutions in C and Python**, by Ed Bueler, is available.

- [Book from SIAM Press](https://my.siam.org/Store/Product/viewproduct/?ProductId=32850137)
- [Google Play E-book](https://play.google.com/store/books/details/Ed_Bueler_PETSc_for_Partial_Differential_Equations?id=tgMHEAAAQBAJ)
:::

:::{admonition} News: New paper on PETSc community
[The Community is the Infrastructure](https://arxiv.org/abs/2201.00967)
:::

## Main Topics

```{toctree}
:maxdepth: 1

overview/index
install/index
tutorials/index
manual/index
manualpages/index
petsc4py/petsc4py
faq/index
community/index
developers/index
miscellaneous/index
```

- [PETSc/TAO Users Manual in PDF](manual/manual.pdf)

(doc-toolkits-use-petsc)=

## Toolkits/libraries that use PETSc

- [ADflow](https://github.com/mdolab/adflow) An Open-Source
  Computational Fluid Dynamics Solver for Aerodynamic and
  Multidisciplinary Optimization
- [BOUT++](https://boutproject.github.io) Plasma simulation
  in curvilinear coordinate systems
- [Chaste](https://www.cs.ox.ac.uk/chaste/) Cancer, Heart and
  Soft Tissue Environment
- [code_aster](https://www.code-aster.org/V2/spip.php?rubrique2​)
  open source general purpose finite element code for solid and
  structural mechanics
- [COOLFluiD](https://github.com/andrealani/COOLFluiD) CFD,
  plasma and multi-physics simulation package
- [DAFoam](https://dafoam.github.io) Discrete adjoint solvers
  with [OpenFOAM](https://openfoam.com) for aerodynamic
  optimization
- [DEAL.II](https://www.dealii.org/) C++ based finite element
  simulation package
- [DUNE-FEM](https://dune-project.org/sphinx/content/sphinx/dune-fem/) Python and C++ based finite element simulation package
- [FEniCS](https://fenicsproject.org/) Python based finite
  element simulation package
- [Firedrake](https://www.firedrakeproject.org/) Python based
  finite element simulation package
- [Fluidity](https://fluidityproject.github.io/) a finite
  element/volume fluids code
- [FreeFEM](https://freefem.org/) finite element PDE solver
  with embedded domain specific language
- [hIPPYlib](https://hippylib.github.io) FEniCS based toolkit
  for solving large-scale deterministic and Bayesian inverse
  problems governed by partial differential equations
- [libMesh](https://libmesh.github.io) adaptive finite element
  library
- [MFEM](https://mfem.org/) lightweight, scalable C++ library
  for finite element methods
- [MLSVM](https://github.com/esadr/mlsvm), Multilevel Support
  Vector Machines with PETSc.
- [MoFEM](http://mofem.eng.gla.ac.uk/mofem/html), An open
  source, parallel finite element library
- [MOOSE - Multiphysics Object-Oriented Simulation
  Environment](https://mooseframework.inl.gov/) finite element
  framework, built on top of libMesh and PETSc
- [OOFEM](http://www.oofem.org) object oriented finite element
  library
- [OpenCarp](https://opencarp.org/) Cardiac Electrophysiology Simulator
- [OpenFOAM](https://develop.openfoam.com/modules/external-solver)
  Available as an extension for linear solvers for OpenFOAM
- [OpenFVM](http://openfvm.sourceforge.net/) finite volume
  based CFD solver
- [PermonSVM](http://permon.vsb.cz/permonsvm.htm) support
  vector machines and
  [PermonQP](http://permon.vsb.cz/permonqp.htm) quadratic
  programming
- [PetIGA](https://bitbucket.org/dalcinl/petiga/) A framework
  for high performance Isogeometric Analysis
- [PHAML](https://math.nist.gov/phaml/) The Parallel
  Hierarchical Adaptive MultiLevel Project
- [preCICE](https://www.precice.org) - A fully parallel
  coupling library for partitioned multi-physics simulations
- [PyClaw](https://www.clawpack.org/pyclaw/) A massively
  parallel, high order accurate, hyperbolic PDE solver
- [SLEPc](https://slepc.upv.es/) Scalable Library for
  Eigenvalue Problems

(doc-index-citing-petsc)=

## Citing PETSc

You can run any PETSc program with the option `-citations` to print appropriate citations for the algorithms you are using within PETSc.

For general citations on PETSc please use the following:

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@misc{petsc-web-page'
```

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@techreport{petsc-user-ref'
```

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@inproceedings{petsc-efficient'
```

For petsc4py usage please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@article{DalcinPazKlerCosimo2011'
```

For PETSc usage on GPUs please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: author
:language: none
:start-at: '@article{MILLS2021'
```

For PetscSF -- parallel communication in PETSc -- please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: pages
:language: none
:start-at: '@article{PetscSF2022'
```

If you use the TS component of PETSc please cite the following:

```{literalinclude} petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@techreport{AbhyankarEtAl2018'
```

If you utilize the TS adjoint solver please cite

```{literalinclude} /petsc.bib
:append: '}'
:end-at: year
:language: none
:start-at: '@article{Zhang2022tsadjoint'
```
