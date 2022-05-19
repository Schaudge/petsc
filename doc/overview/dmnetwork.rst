===================
DMNetwork
===================

PETSc DMNetwork provides a powerful flexible scalable framework for simulation of multiphysics phenomena over large-scale networked systems. It eases the applications development cycle by providing the necessary infrastructure through simple abstractions to define and query the network components.

Some complex systems that could exploit PETSc DMNetwork include critical infrastructures (power, water, gas, transportation, and telecommunicaiton), human body (brain and blood circulation), ecology (foodweb and pollinator-plant), disease (epidemic and virus-host networks) and other social networks.

.. image:: /images/overviews/dmnetwork/network.svg
   :alt: Complex System Examples
   :align: center

PETSc DMNetwork provides data and topology management, parallelization for multiphysics systems over a network, and hierarchical and composable solvers to exploit the problem structure. The key features of DMNetwork include creating the network layout, partitioning for efficient parallelism, parallel data movement, utility routines for extracting connectivity information, and linear, nonlinear, and time-stepping solvers.

The steps for using DMNetwork is illustrated in the figure below; they include: (i) Create network graph, (ii) Add physics components, and variables to the network elements (edges and vertices), (iii) Distribute the populated network to multiple processors, and (iv) decompose the domains and associate them with their respective linear, nonlinear, time-stepping solvers.

.. image:: /images/overviews/dmnetwork/dmnetwork.svg
   :alt: DMNetwork Workflow
   :align: center

Case Studies
============

Case Study 1: Power Flow Simulation
-----------------------------------

A contingency analysis done on U.S. power grid system using an AC power flow model developed using DMNetwork. The input file was obtained from the MatPower package (`matpower.org`_). The total numbers of buses, generators, loads, and branches are 82,000, 13,419, 37,755, and 104,121 respectively. The total number of unknowns solved is around half a million. Increasing the number of cores from 128 to 2048, provided a speedup of twelve. `Poster`_

  .. _matpower.org: https://matpower.org/

  .. _Poster: https://www.mcs.anl.gov/petsc/OLD/dmnetwork/documents/Application1_Betrie_etal-2019_poster.pdf

.. image:: /images/overviews/dmnetwork/power_topology.svg
   :alt: Power Network Topology
   :align: center

.. image:: /images/overviews/dmnetwork/power_scaling.svg
   :alt: Power Network Topology
   :align: center

Case Study 2: River Flow Simulation
-----------------------------------

A river flow simulation done on the U.S. river networks using DMNetwork. The input file was obtained from the NHDPlus dataset (`horizon-systems.com`_). The total numbers of reaches and junctions are 3,098,638 and 3,036,092, respectively. The total number of unknowns solved is around half a billion. Increasing the number of cores from 1,024 to 65,536, provided a speedup of 35. `Poster`_

  .. _horizon-systems.com: https://www.horizon-systems.com/

  .. _Poster: https://www.mcs.anl.gov/petsc/OLD/dmnetwork/documents/Application1_Betrie_etal-2019_poster.pdf

.. image:: /images/overviews/dmnetwork/river_topology.svg
   :alt: River Flow Simulation
   :align: center

.. image:: /images/overviews/dmnetwork/river_scaling.svg
   :alt: River Flow Simulation
   :align: center

Tutorials
=========

Example 1: Electric Circuit
---------------------------

This example demonstrates simulation of a linear problem of electric circuit using the DMNetwork interface. Further details: `KSP example 1 <../../src/ksp/ksp/tutorials/network/ex1.c.html>`_.

* Compile ex1.c
            cd petsc/src/ksp/ksp/tutorials/network
            make ex1

* Run a 1 processor example and view solution at edges and vertices
            mpiexec -n 1 ./ex1

* Run a 1 processor example with a convergence reason
            mpiexec -n 1 ./ex1 -ksp_converged_reason

* Run with 2 processors with a partitioning option
            mpiexec -n 2 ./ex1  -petscpartitioner_type simple

Example 2: AC Power Flow
------------------------

This example demonstrates simulation of a nonlinear power flow in a grid network using the DMNetwork interface. Further details: `SNES example power <../..src/snes/tutorials/network/power/power.c.html>`_.

* Compile power.c
            cd petsc/src/snes/tutorials/network/power
            make power

* Run a 1 processor example and view solution at vertices
            mpiexec -n 1 ./power

* Run with 2 processors with edge and vertex visualization
            mpiexec -n 2 ./power  -dm_view

Example 3: Water Flow in pipes
------------------------------
This example demonstrates simulation of a transient water flow in a pipe network using the DMNetwork interface. Further details: `TS example pipe <../..src/ts/tutorials/network/wash/pipes1.c.html>`_.

* Compile pipes1.c
            cd petsc/src/ts/tutorials/network/wash
            make  pipes1

* Run with 2 processors with a partitioning option
            mpiexec -n 2 ./pipes1 -ts_monitor -case 1 -ts_max_steps 1 -petscpartitioner_type
            simple -options_left no -viewX

* Run with 3 processors with a different case and more time-stepping options
            mpiexec -n 3 ./pipes1  -ts_monitor -case 2 -ts_max_steps 10 -petscpartitioner_type
            simple -options_left no -viewX

More Examples
-------------

* `SNES example 1 <../..src/snes/tutorials/network/ex1.c.html>`_
* `SNES example water <../..src/snes/tutorials/network/water/water.c.html>`_

Publications
============

Overview Materials:
-------------------

* Abhyankar S., Betrie G., Maldonado D.A, McInnes L.C., Smith B., Zhang H. (2020). `PETSc DMNetwork`_: A Library for Scalable Network PDE-Based Multiphysics Simulations. ACM Transactions on Mathematical Software, Vol. 46, No.1, Article 5, 2020.

   .. _PETSc DMNetwork: https://doi.org/10.1145/3344587

Applications:
-------------

* Betrie G., Simith B., Zhang H. (2019). A Scalable Multiphysics Modeling Package For Critical Networked Infrastructures Using PETSc DMNetwork. Argonne Postdoctoral Research and Career Symposium, Lemont, IL, 7 November 2019. Application: `Critical Infrastructure`_

  .. _Critical Infrastructure: https://www.mcs.anl.gov/petsc/OLD/dmnetwork/documents/Application1_Betrie_etal-2019_poster.pdf

* Betrie G., Zhang H., Simith B., Yan E. (2018). A Scalable River Network Simulator For Extereme Scale Computers Using the PETSc Library. AGU Fall Meeting, Washington, D.C., 10-14 December 2018. Application: `River Flow Simulation`_

  .. _River Flow Simulation: https://www.mcs.anl.gov/petsc/OLD/dmnetwork/documents/Application2_Betrie_etal_2018_slide.pdf

* Werner A., Duwadi K., Stegmeier N., Hansen T., Kimn J. (2019). Parallel Implementation of AC Optimal Power Flow and Time Constrained Optimal Power Flow Using High Performance Computing.IN IEEE 9th Annual Computing and Communication Workshop and Conference. Application: `Optimal Power Flow Simulation`_

  .. _Optimal Power Flow Simulation: https://ieeexplore.ieee.org/document/8666551

* Rinaldo S., Ceresoli A., Lahaye D., Merlo M., Cvetkovic M., Vitiello S., Fulli G. (2018). Distributing Load Flow Computations Across System Operators Boundaries Using the Newton-Krylov-Schwarz Algorithm Implemented in PETSc. Application: `Power Flow Simulation`_

  .. _Power Flow Simulation: https://www.mdpi.com/1996-1073/11/11/2910
