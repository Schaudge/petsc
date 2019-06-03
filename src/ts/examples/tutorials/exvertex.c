static char help[] = "Test getting all vertices in mesh";

#include <petscdmplex.h>
#include <petscviewer.h>

#define PETSCVIEWERASCII 	"ascii"


int main(int argc, char **argv)
{
        PetscErrorCode		ierr;
        MPI_Comm                comm;
        DM                      dm, dmDist;
        PetscBool		dmInterp = PETSC_TRUE;
        IS			bcPointsIS;
        PetscInt		dStart, dEnd, i, counter = 0, numBC, nVertex, nCoords;
        PetscScalar		*vecArray, *coordArray;
        const PetscInt		numComp[1], numDOF[1], bcField[1];
        Vec			locVec, coords;
        PetscViewer		viewer;



        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
        comm = PETSC_COMM_WORLD;
        ierr = PetscViewerCreate(comm, &viewer);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);
        ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INDEX);CHKERRQ(ierr);
        ierr = DMPlexCreateFromFile(comm, "2Drectq4.exo", dmInterp, &dm);CHKERRQ(ierr);


        numFields = 1;
        numComp[0] = 1;
        numDOF[0] = 1;
        numBC = 1;
        bcField[0] = 0;

        
        ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
        /*	Get Vertices	*/
        ierr = DMPlexGetDepthStratum(dm, 0, &dStart, &dEnd);CHKERRQ(ierr);
        ierr = DMGetLocalVector(dm, &locVec);CHKERRQ(ierr);
        ierr = VecGetLocalSize(locVec, &nVertex);CHKERRQ(ierr);
        ierr = VecGetArray(locVec, &vecArray);CHKERRQ(ierr);

        
        ierr = DMGetCoordinatesLocal(dm, &coords);
        ierr = VecGetLocalSize(coords, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);
        printf("no error yet\n");
        ierr = VecView(locVec, viewer);CHKERRQ(ierr);
        for (i = dStart; i < dEnd; i++) {
                //                printf("%d\n", counter);
                counter++;
        }



        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
