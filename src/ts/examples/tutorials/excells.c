static char help[] = "Test getting all cells in mesh";

#include <petscdmplex.h>
#include <petscviewer.h>

#define PETSCVIEWERASCII 	"ascii"


int main(int argc, char **argv)
{
        PetscErrorCode		ierr;
        MPI_Comm                comm;
        DM                      dm;
        PetscSection		section;
        PetscBool		dmInterp = PETSC_TRUE;
        IS			verts, cells, bcPointsIS;
        PetscInt		dim = 2, vStart, vEnd, j, counter = 0, numFields, numBC, numcells, numindices, *indices, offset;
        PetscInt		ic, cStart, cEnd;
        PetscScalar		*coordArray;
        PetscInt		numComp[1], numDOF[3], bcField[1];
        const PetscInt		*vertids, *cellids;
        Vec			coords;
        PetscViewer		viewer;



        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
        comm = PETSC_COMM_WORLD;
        ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INDEX);CHKERRQ(ierr);
        //ierr = DMPlexCreateFromFile(comm, "2D1x1.exo", dmInterp, &dm);CHKERRQ(ierr);
        //ierr = DMPlexCreateFromFile(comm, "3Dbrick.exo", dmInterp, &dm);CHKERRQ(ierr);
        //ierr = DMPlexCreateFromFile(comm, "3Dbrick4els.exo", dmInterp, &dm);CHKERRQ(ierr);
        ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, dmInterp, &dm);

        numFields = 1;
        numComp[0] = 1;
        for (PetscInt k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
        numDOF[0] = 1;
        numBC = 0;

        // Please note that bcField stays uninitialized because numBC = 0,
        // therefore having a trash value. This is probably handled internally
        // within DMPlexCreateSection but idk how exactly.
        ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
        ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
        ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetSection(dm, section);CHKERRQ(ierr);

        /*	Get Cells	*/
        ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
        ierr = DMGetStratumIS(dm, "depth", 0, &verts);CHKERRQ(ierr);
        ierr = ISGetIndices(verts, &vertids);CHKERRQ(ierr);

        // OANA: Why was the index 3 instead of 2 here? I thought it only went from 0-2...
        ierr = DMPlexGetDepthStratum(dm, dim /*3*/, &cStart, &cEnd);CHKERRQ(ierr);
        ierr = DMGetStratumIS(dm, "depth", dim /*3*/, &cells);CHKERRQ(ierr);
        ierr = ISGetIndices(cells, &cellids);CHKERRQ(ierr);

        /*	Get Local Coordinates	*/
        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);

        offset=cEnd-cStart;
 	ierr = PetscPrintf(comm, " Total number vertices %d\n", vEnd-vStart);CHKERRQ(ierr);
        for (ic = 0; ic < cEnd-cStart; ic++)
	{ 	ierr = DMPlexGetClosureIndices(dm,section,section,cellids[ic],&numindices,&indices,NULL);CHKERRQ(ierr);
		ierr = DMPlexGetConeSize(dm, cellids[ic], &numcells);
                ierr = PetscPrintf(comm, "Current cell %d, total number %d  \n", ic, offset);CHKERRQ(ierr);
                for (j = 0; j < numindices; j++){
                        ierr = PetscPrintf(comm, "x(%2d, %2d, %2d)=(%.2f,%.2f,%0.2f)   \n", dim*(indices[j]), (dim*(indices[j]))+1, (dim*(indices[j]))+2,
                                           coordArray[dim*(indices[j])], coordArray[(dim*(indices[j]))+1], coordArray[(dim*(indices[j]))+2]);CHKERRQ(ierr);
                }
                ierr = DMPlexRestoreClosureIndices(dm,section,section,ic,&numindices,&indices,NULL);CHKERRQ(ierr);
                ierr = PetscPrintf(comm, "      \n");CHKERRQ(ierr);
                counter++;
        }

	ierr = VecRestoreArray(coords,&coordArray);CHKERRQ(ierr);

        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
