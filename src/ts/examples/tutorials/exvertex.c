static char help[] = "Test getting all vertices in mesh";

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
        IS			points, bcPointsIS;
        PetscInt		dim = 3, dStart, dEnd, i, counter = 0, numFields, numBC;
        PetscScalar		*coordArray;
        PetscInt		numComp[1], numDOF[1], bcField[1];
        const PetscInt		*idx;
        Vec			coords;
        PetscViewer		viewer;



        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
        comm = PETSC_COMM_WORLD;
        ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INDEX);CHKERRQ(ierr);
        ierr = DMPlexCreateFromFile(comm, "2D1x1.exo", dmInterp, &dm);CHKERRQ(ierr);
        ierr = DMPlexCreateFromFile(comm, "3Dbrick.exo", dmInterp, &dm);CHKERRQ(ierr);
        //ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, dmInterp, &dm);

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

        /*	Get Vertices	*/
        ierr = DMPlexGetDepthStratum(dm, 0, &dStart, &dEnd);CHKERRQ(ierr);
        ierr = DMGetStratumIS(dm, "depth", 0, &points);CHKERRQ(ierr);
        ierr = ISGetIndices(points, &idx);CHKERRQ(ierr);

        /*	Get Local Coordinates	*/
        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);

        ierr = PetscPrintf(comm, "Vertex Num |     Coord     | IS Index\n");CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "-------------------------------------\n");CHKERRQ(ierr);
        for (i = 0; i < dEnd-dStart; i++) {
                ierr = PetscPrintf(comm, "     %d     | (%.2f , %.2f) | %d\n", counter+1, coordArray[2*i], coordArray[(2*i)+1], idx[counter]);CHKERRQ(ierr);
                counter++;
        }

        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
