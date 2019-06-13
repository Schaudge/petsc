static char help[] = "Test getting all edges in mesh";

#include <petscdmplex.h>
#include <petscviewer.h>

#define PETSCVIEWERASCII        "ascii"


int main(int argc, char **argv)
{
        PetscErrorCode          ierr;
        MPI_Comm                comm;
        DM                      dm, dmDist;
        PetscSection            section;
        PetscBool               dmInterp = PETSC_TRUE;
        IS                      faces, bcPointsIS;
        PetscInt                dim = 2, dOffset, i, j, counter = 0, numFields, numBC, numFaces;
        PetscScalar             *coordArray;
        PetscInt                faceDim[2], numComp[1], numDOF[1], bcField[1];
        const PetscInt          *faceidx;
        Vec                     coords;
        PetscViewer             viewer;



        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
        comm = PETSC_COMM_WORLD;

        ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INDEX);CHKERRQ(ierr);

	faceDim[0] = 10;
	faceDim[1] = 10;
	//ierr = DMPlexCreateFromFile(comm, "2D1x1.exo", dmInterp, &dm);CHKERRQ(ierr);
        ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faceDim, NULL, NULL, NULL, dmInterp, &dm);

	ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		dm = dmDist;
	}
        numFields = 1;
        numComp[0] = 1;
        numDOF[0] = 1;
        numBC = 0;
        // Please note that bcField stays uninitialized because numBC = 0,
        // therefore having a trash value. This is probably handled internally
        // within DMPlexCreateSection but idk how exactly.

        ierr = DMGetStratumIS(dm, "Cell Sets", 0, &bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
        ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
        ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetSection(dm, section);CHKERRQ(ierr);

        /*      Get Edges    */
        // This call is purely to get the numerical offset "n" within the IS due to the cell IS.
        // This is because for whatever fantastical reason the internal order is:
        // [[cell ids (id0 to idn)],[point ids (idn+1 to idm)],[face ids (idm+1 to idEnd)]]
        // but we need to index from 0 (i.e. IS entry n+1) in the actual coordinate array below
        ierr = DMPlexGetDepthStratum(dm, 2, NULL,  &dOffset);CHKERRQ(ierr);
        ierr = DMGetStratumIS(dm, "depth", 1, &faces);CHKERRQ(ierr);
        ierr = ISGetSize(faces, &numFaces);CHKERRQ(ierr);
        ierr = ISGetIndices(faces, &faceidx);CHKERRQ(ierr);

        /*      Get Local Coordinates   */
        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);

        ierr = PetscPrintf(comm, "Edge Num |          Coord          | IS Index\n");CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "---------------------------------------------\n");CHKERRQ(ierr);
        for (i = 0; i < numFaces; i++) {
                const PetscInt	curFace = faceidx[i];
                const PetscInt	*curPoints;
                PetscInt	numPoints; // Number of points to define the face, for 1D = 2;
                ierr = PetscPrintf(comm, "    %2d     ", counter+1);CHKERRQ(ierr);
                ierr = DMPlexGetCone(dm, curFace, &curPoints);
                ierr = DMPlexGetConeSize(dm, curFace, &numPoints);
                for (j = 0; j < numPoints; j++){
                        ierr = PetscPrintf(comm, "(%.2f,%.2f)", coordArray[2*(curPoints[j]-dOffset)], coordArray[(2*(curPoints[j]-dOffset))+1]);CHKERRQ(ierr);
                        if (j < numPoints-1){
                                ierr = PetscPrintf(comm, "->");CHKERRQ(ierr);
                        }
                }
                ierr = PetscPrintf(comm, "      ");CHKERRQ(ierr);
                ierr = PetscPrintf(comm, "%d\n", curFace);CHKERRQ(ierr);
                counter++;
        }
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
