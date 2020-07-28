TUTORIAL: DMPlex
================

:xref:`DMPlex` is used within PETSc to handle unstructured meshes with useful interfaces
for both topology and geometry. A :xref:`DMPlex` object is based on defining a mesh as a
Hasse Diagram, where every level of topology is a separate layer in the diagram. This
necessarily allows for easy definition of the inherent links between topological groups.

Topology
********

:xref:`DMPlex` employs two structures to define a certain point, namely the points'
"height" and "depth".

.. graphviz::

   digraph G {
      d -> c -> b -> a [dir=back];
      e -> f -> g -> h;
      a [label="Depth 0: Vertices"];
      b [label="Depth 1: Edges"];
      c [label="Depth 2: Faces"];
      d [label="Depth 3: Cells"];
      e [label="Height 0: Cells"];
      f [label="Height 1: Faces"];
      g [label="Height 2: Edges"];
      h [label="Height 3: Vertices"];
   }

:xref:`DMPlex` not only needs to know about a points topological identity, but also about
its location within the greater graph. Given that :xref:`DMPlex` objects handle
unstructured meshes, every point is defined by its "cone" and "support/closure".

* The support/closure of a point is defined as the set of all point of the next highest
  topological dimension that is connected to the current point. For example, in a 3
  dimensional *fully interpolated* mesh, the closure of a vertex would be the set of all
  of the edges which contain this vertex.

* The cone of a point is defined as the set of every points of the next lowest topological
  dimension. For example, in a 3 dimensional *fully interpolated* mesh, the cone of a cell would
  be the set of every face on belonging to that cell.

Interpolation
*************

Interpolation in :xref:`DMPlex` is defined as a mesh which contains some but not all
"intermediate" entities. Normally a user will have a *fully interpolated* mesh (i.e. a
mesh where 3 dimensional cells have faces, edges, and vertices) and so an interpolated
mesh behaves as expected. :xref:`DMPlex` however can also handle *partially interpolated*
meshes where any of the intermediate entities may be missing (although not all
:xref:`DMPlex` functions support this). See :xref:`DMPlexInterpolate`,
:xref:`DMPlexIsInterpolated`, :xref:`DMPlexIsInterpolatedCollective`.

This tutorial will show you how to:

* Create and setup a simple 2D box tensor-grid mesh

* Retrieve and alter local values

Creating A DMPlex Object
************************

As per usual, every PETSc program must should start with a ``help[]`` message, and any
``include`` directives that you may need. In this case, only ``petscdmplex.h`` is
necessary.

.. code-block:: c

   static char help[] = "Using DMPlex in PETSc.\n\n";

   # include <petscdmplex.h>

   int main(int argc, char **argv)
   {
	PetscErrorCode	ierr;
	MPI_Comm        comm;
	DM              dm, dmDist;
	PetscSF		distributionSF;
	PetscInt        i, dim = 2, overlap = 0;
	PetscInt        faces[dim];
	PetscBool       simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

	ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
	comm = PETSC_COMM_WORLD;

	for (i = 0; i < dim; i++) {
		faces[i] = 2;
	}

	ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, /* lower */ NULL, /* upper */ NULL, /* periodicity */ NULL, dmInterped, &dm);CHKERRQ(ierr);

	ierr = DMPlexDistribute(dm, overlap, &distributionSF, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		dm = dmDist;
	}

In this case a 2 dimensional square grid is generated on a [0, 1] X [0, 1] box with 2
faces per edge, resulting a 2x2 cell grid. :xref:`DMPlex` also generates the intermediate
edge information due to passing ``dmInterped = PETSC_TRUE`` into the original creation
routine.

Once created, the :xref:`DMPlex` object is then distributed to all available
processses. At this point, each process checks that its distributed section of
the :xref:`DMPlex` object is not ``NULL``, and replaces its non distributed :xref:`DMPlex`
object with the distributed one.

Setting Up Internal PetscSection
********************************

In this next step, the core of the internal accounting of the :xref:`DMPlex` object is set
up by creating a PetscSection describing the layout of the various fields, boundary
conditions, and degrees of freedoms.

.. code-block:: c

   PetscInt	 numFields = 1, numBC = 1;
   PetscInt	 numDOF[numFields*(dim+1)], numComp[numFields], bcField[numBC];
   IS            bcPointsIS;
   PetscSection  section;

   /*	Number of Components per Field	*/
   for (i = 0; i < numFields; i++) { numComp[i] = 1;}

   /*	Spatial DOF of field component (0 to not use)	*/
   for (i = 0; i < numFields*(dim+1); i++) { numDOF[i] = 0;}

   /*	Specificy DOF of field component that we intend to use	*/
   numDOF[0] = 1;

   /*	Boundary condition of field component set to Dirichlet	*/
   bcField[0] = 0;

   ierr = DMGetStratumIS(dm, "depth", dim, &bcPointsIS);CHKERRQ(ierr);
   ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
   ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
   ierr = PetscSectionSetFieldName(section, 0, "Default_Field");CHKERRQ(ierr);
   ierr = DMSetSection(dm, section);CHKERRQ(ierr);

   /*	Dont forget to clean up	*/
   ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
   ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);

After initializing the necessary meta data, you must specificy to what topological entity
you wish to assign the values. In this case, we assign our boundary condition of 0 onto
the cells. Recall that a depth of 0 corresponds to vertices, whereas a depth equal to the
dimension of a *fully interpolated* mesh will return the "cells" (technically these are faces,
however for ease-of-understanding they shall be henceforth referred to as cells).

This information is then used to generate a complete PetscSection object, and assigned to the
:xref:`DMPlex`.

Performing A Simple Operation on Vertices
*****************************************

Finally, we are ready to perform calculations on the :xref:`DMPlex`. In this example we
will retrieve a vector containing the values on the vertex coordinates, and add a scalar to them
to illustrate this concept.

.. code-block:: c

   Vec		vec;
   PetscScalar  *vecArray;
   PetscInt     vecSize, j;

   ierr = DMGetLocalVector(dm, &vec);CHKERRQ(ierr);
   ierr = VecGetSize(vec, &vecSize);CHKERRQ(ierr);

   /*    Can pass 0 in place of viewer, to print to STDOUT	*/
   ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

   ierr = VecGetArray(vec, &vecArray);CHKERRQ(ierr);
   for (j = 0; j < vecSize; j++) {
     vecArray[j] = 5;
   }
   ierr = VecRestoreArray(vec, &vecArray);CHKERRQ(ierr);

   ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = DMRestoreLocalVector(dm, &vec);CHKERRQ(ierr);

.. important::

   It is important that you do not destroy the retrieved :xref:`Vec` here.
   :xref:`DMGetLocalVector` does not create a vector, but merely provides a pointer to an
   existing vector created during the section setup process!

Cleanup
*******

Finally, we free memory that was used during our program and call PetscFinalize
which finalizes PETSc libraries and MPI.

.. code-block:: c

   ierr = DMDestroy(&dm);CHKERRQ(ierr);
   ierr = PetscFinalize();CHKERRQ(ierr);
   return ierr;
   }


Running
*******

Use the command line to compile, and run your program. Depending on whether or not you
wish to run the program in parallel, you may specifcy so at this time.

.. code:: bash

   ~$ make your_dmplex_program
   ~$ ./your_dmplex_program # Run in serial
   Vec Object: 1 MPI processes
   type: seq
   0.
   0.
   0.
   0.
   0.
   0.
   0.
   0.
   0.
   Vec Object: 1 MPI processes
   type: seq
   5.
   5.
   5.
   5.
   5.
   5.
   5.
   5.
   5.
   ~$ mpiexec -n 2 ./your_dmplex_program # Run in parallel!
   Vec Object: 1 MPI processes
   type: seq
   0.
   0.
   0.
   0.
   0.
   0.
   Vec Object: 1 MPI processes
   type: seq
   5.
   5.
   5.
   5.
   5.
   5.

.. note::

   Note that with more than 1 process, less of the :xref:`Vec` is printed out as each process
   will only print out its local section of the :xref:`Vec`.
