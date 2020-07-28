==========================
DM: Grid/Domain Management
==========================

PETSc hands-on tutorial: DM Basics
==================================

Like any solver suite worth its salt, PETSc must be able to interface with, and perform a
suite of general use functions on discretized meshes. For these purposes PETSc uses the Domain
Management (:xref:`DM`) arm of the library, which is split into several sub-sections each focusing
on a different kind of framework. These subsections are:


While each :xref:`DM` implementation is generally tailored towards a certain problem approach, all
methods are rooted in the :xref:`DM` object class. For further reading please see:

* :xref:`DMComposite`

* :xref:`DMDA`

* :xref:`DMForest`

* :xref:`DMMOAB`

* :xref:`DMNetwork`

* :xref:`DMPatch`

* :xref:`DMPlex`

* :xref:`DMStag`

* :xref:`DMSwarm`
