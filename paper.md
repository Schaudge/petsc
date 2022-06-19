---
title: 'DMStag: Staggered, Structured Grids for PETSc'
tags:
  - C
authors:
  - name: Patrick Sanan
    orcid: 0000-0003-3968-8482
    affiliation: "1, 2"
  - name: Dave A. May
    orcid:  0000-0003-2471-7498
    affiliation: "3"
  - name: Richard T. Mills
    orcid: 0000-0003-0683-6899
    affiliation: "2"
  - name: Boris J. P. Kaus
    orcid: 0000-0002-0247-8660
    affiliation: "4"
affiliations:
 - name: ETH Zurich, Switzerland
   index: 1
 - name: Argonne National Laboratory, USA
   index: 2
 - name: University of California, San Diego
   index: 3
 - name: Johannes Gutenberg University Mainz, Germany
   index: 4
date: 9 June 2022
bibliography: paper.bib
---

# Summary

"Staggered grid" codes arise in many applications, where a physical problem is discretized and solved in terms of unknowns located at different locations on a structured grid. For example, the Navier-Stokes and Stokes equations can be solved on a staggered grid with the classical Marker and Cell (MAC) method [@HarlowWelch1965], storing pressure unknowns in element centers and velocities on element boundaries. This gives a simple and compact method which exhibits superior numerical properties to colocated grid methods which consider all unknowns to be located on the same grid of points.
DMStag is a component of [PETSc](https://petsc.org) [@BalayEtAl2022;@BalayEtAl2022b], the Portable, Extensible Toolkit for Scientific Computation, an MPI-parallel C library which is widely used to perform very large simulations in computational science and engineering. DMStag is a new implementation of the `DM` (Domain Management) class to provide a native abstraction for defining and solving partial differential equations (PDEs) on structured, staggered grids, in parallel . With DMStag, one can write code which will run on any number of MPI ranks and rely on the `DM` API to work with global vectors representing the full discretized problem, and with regularly-blocked local vectors on the overlapping local patch upon which lower-level computations are performed. This allows users to scale codes to thousands of processors, and provides a platform upon which to compose the advanced, scalable PDE solvers, and to leverage various computational backends, notably including GPUs [@MillsEtAl2021].

DMStag provides a simple API, similar to that for the DMDA class, to allow a user to create and interact with objects which (like all `DM` objects) represent

  1. A discrete topological space (here, a structured cell complex)
  2. An *atlas* of overlapping local patches, assigned to MPI ranks
  3. A *field* assigning sets of scalar unknowns to each point in the topological space (here, a constant number of unknown "DOFs" for each point in a given *stratum*, defined as cells of a given dimension)
  4. A special field for coordinates of each point

For more information, please consult the [PETSc manual chapter on DMStag](https://petsc.org/main/docs/manual/dmstag/) and the [DMStag manual pages](https://petsc.org/main/docs/manualpages/DMSTAG).

# Statement of need

The discretization of PDEs on structured (or "regular") grids of cells, associated unknown quantities with several different geometric entities (for instance, cell centers and cell faces), has applications in many fields. The highly structured nature of this discretization and relative simplicity of many of the stencils can lead to highly efficient and maintainable application codes which solve important problems in fluid simulation (including weather and climate simulation and geophysical flows) and electromagnetic and plasma simulations. Users have commonly implemented these methods with PETSc, using the DMDA object. However, since this represents a structured grid with an equal number of unknowns attached to each vertex (or cell center, if interpreted that way), a staggered grid must either be represented as a collection of DMDA objects, or by introducing a convention to store staggered data at a nearby non-staggered location, ignoring unused points on the boundary. This is inconvenient and can also have performance consequences. The choice of approach commits one to interlace or segregate data from different physical fields, and the presence of unused "dummy" points in the global representation may induce additional complication when attempting to use grid-aware solvers like PETSc's geometric or algebraic multigrid solvers, or block-based preconditioners (`PCFIELDSPLIT`).

DMStag is already in use in ongoing research simulating magmatic flow in the Earth's mantle, and MHD simulations for tokamak simulation. Its base functionality was introduced into PETSc with version 3.11 (March 2019).

We are not aware of any directly-comparable general-purpose frameworks. In many application codes, the relatively simple data structures involved can be implemented directly. This approach has the obvious advantage of avoiding a heavy dependency like PETSc, but could incur larger investment of implementation effort to experiment with new, more-scalable solvers or with GPU-backed data structures, both of which PETSc's focus on composability and portability aim to make more accessible. Projects like [GridTools](https://github.com/GridTools/gridtools) and STELLA [@GysiEtAl2015] provide high-performance stencil operations for particular domains, in this case climate and weather modelling.


# Examples

A few example codes are available with PETSc, currently in `src/dm/impls/stag/tutorials`. Additional codes in `src/dm/impls/stag/tests` are used for testing.

Example `ex6` modifies a simple velocity-stress formulation for seismic wave propagation [@Virieux1986] to operate in 3D: this shows usage of much of the basic API of DMStag to perform explicit time-stepping of fields associated with multiple, compatible DMStag objects. \autoref{fig:ex6} shows one component of the velocity from one timestep of a 2D simulation with this code.

![$y$ velocity for a single timestep of a seismic wave simulation generated with DMStag `ex6`.\label{fig:ex6}](https://github.com/psanan/dmstag_joss_temp/raw/main/dmstag_joss_paper_images/ex6.png)

Example `ex4` solves the variable-viscosity, stationary Stokes equations with finite differences (see e.g. @Gerya2019). This demonstrates assembling and solving an explicit linear system. \autoref{fig:ex4} shows a flow field generated with this code.

![Flow field for stationary Stokes flow around a cubic, high-viscosity inclusion, inspired by a benchmark setup from @FuruichiMayTackley2011 and computed with DMStag tutorial `ex4`.\label{fig:ex4}](https://github.com/psanan/dmstag_joss_temp/raw/main/dmstag_joss_paper_images/ex4.png)

# Acknowledgements

We thank the many members of the PETSc developer and user community who have contributed in ways large and small to this code's development and maintenance.

P. S. acknowledges financial support from the Swiss University Conference and the Swiss Council of Federal Institutes of Technology through the Platform for Advanced Scientific Computing (PASC) program, as part of the StagBL project.

# References
