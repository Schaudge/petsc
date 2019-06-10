 static char help[] = "Diagnose mesh problems";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscmath.h>

# define PETSCVIEWERASCII        "ascii"
# define PETSCVIEWERVTK          "vtk"
# define MATAIJ             	 "aij"
/*	2D Array Routines	*/
PetscErrorCode StretchArray2D(DM dm, PetscScalar lx, PetscScalar ly)
{
        /*
         Analytical Jacobian:
         J = lx*ly
        */


        PetscErrorCode		ierr;
        PetscInt		i, nCoords;
        Vec			coordsLocal;
        PetscScalar		*coordArray;

        ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

        // Order in coordarray is [x1,y1,z1....]
        for (i = 0; i < nCoords; i++) {
                if (i % 2) {
                        coordArray[i-1] = lx*coordArray[i-1];
                        coordArray[i] = ly*coordArray[i];
                }
        }

        ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coordsLocal);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode ShearArray2D(DM dm, PetscScalar theta)
{
        /*
         Analytical Jacobian:
         J = cos(theta
        */


        PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsLocal;
        PetscScalar             *coordArray;

        ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

        // Order in coordarray is [x1,y1,z1....]
        for (i = 0; i < nCoords; i++) {
                if (i % 2) {
                        coordArray[i-1] = coordArray[i-1] + coordArray[i]*PetscSinReal(theta);
                        coordArray[i] = coordArray[i]*PetscCosReal(theta);
                }
        }

        ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coordsLocal);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode SkewArray2D(DM dm, PetscScalar omega)
{
        /*
         Analytical Jacobian:
         J = cos(omega)
        */


        PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsLocal;
        PetscScalar             *coordArray;

        ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

        // Order in coordarray is [x1,y1,z1....]
        for (i = 0; i < nCoords; i++) {
                if (i % 2) {
                        coordArray[i] = coordArray[i] + coordArray[i-1]*PetscSinReal(omega);
                        coordArray[i-1] = coordArray[i-1]*PetscCosReal(omega);
                        // reversing order sice "y" is changed first
                }
        }

        ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coordsLocal);CHKERRQ(ierr);

        return ierr;
}

PetscErrorCode LargeAngleDeformArray2D(DM dm, PetscScalar phi)
{
        /*
         Analytical Jacobian:
         J = 1 + phi*(eta - nu)
         where eta and nu are the omega* coordinates not regular x and y
        */


        PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsTemp, coordsLocal;
        PetscScalar		mx, my, bx, by;
        PetscScalar             *tempCoordArray, *coordArray;

        if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }
        ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
        ierr = DMCreateLocalVector(dm, &coordsTemp);CHKERRQ(ierr);
        ierr = VecGetArray(coordsTemp, &tempCoordArray);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

        // Order in coordarray is [x1,y1,z1....]
        for (i = 0; i < nCoords; i++) {
                if (i % 2) {
                        mx = (1 - (2*coordArray[i-1]))*(phi/2);
                        my = -(1 - (2*coordArray[i]))*(phi/2);
                        bx = -(1 - (2*coordArray[i-1]))*(phi/4);
                        by = (1 - (2*coordArray[i]))*(phi/4);
                        tempCoordArray[i-1] = coordArray[i-1] + (mx*coordArray[i]) + bx;
                        tempCoordArray[i] = coordArray[i] + (my*coordArray[i-1]) + by;
                        // need temp because both entries are changed
                }
        }

        ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
        ierr = VecRestoreArray(coordsTemp, &tempCoordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coordsTemp);CHKERRQ(ierr);
        ierr = VecDestroy(&coordsTemp);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode SmallAngleDeformArray2D(DM dm, PetscScalar phi)
{
	/*
         Analytical Jacobian:
         J = 1 - phi*(2*eta - 1)
         where eta and nu are the omega* coordinates not regular x and y
        */


        PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsLocal;
        PetscScalar             my, by;
        PetscScalar             *coordArray;

        if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }
        ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
        ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

        // Order in coordarray is [x1,y1,z1....]
        for (i = 0; i < nCoords; i++) {
                if (i % 2) {
                        my = (1 - (2*coordArray[i]))*phi;
                        by = -(1 - (2*coordArray[i]))*(phi/2);
                        coordArray[i] = coordArray[i] + (my*coordArray[i-1]) + by;
                        // x = eta in this case
                }
        }
        ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coordsLocal);CHKERRQ(ierr);
        return ierr;
}

/*	2D Jacobians	*/
PetscErrorCode Stretch2DJacobian(DM dm, PetscScalar lx, PetscScalar ly)
{
        PetscErrorCode	ierr;
        IS		pointsIS;
        Mat		J;
        const PetscInt	*pointsidx;
        PetscInt	ISSize;
        PetscScalar     arrayJ[1], val;
        PetscViewer	matview;

        ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
        ierr = MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        ierr = DMGetStratumIS(dm, "depth", 0, &pointsIS);CHKERRQ(ierr);
        ierr = ISGetIndices(pointsIS, &pointsidx);CHKERRQ(ierr);
        ierr = ISGetSize(pointsIS, &ISSize);CHKERRQ(ierr);
        val = lx*ly;
        printf("%d\n",pointsidx[0]-4);
        for (PetscInt i = 0; i < ISSize; i++) {
                for (PetscInt j = 0; j < ISSize; j++) {
                        ierr = MatSetValuesLocal(J, 1, &i, 1, &j, &val, INSERT_VALUES);
                }
        }
        ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
        ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
        ierr = PetscViewerCreate(PETSC_COMM_WORLD, &matview);
        ierr = PetscViewerSetType(matview, PETSCVIEWERASCII);
        ierr = PetscViewerPushFormat(matview, PETSC_VIEWER_ASCII_DENSE);
        MatView(J, 0);

        return ierr;
}

int main(int argc, char **argv)
{
        PetscErrorCode          ierr;
        MPI_Comm                comm;
        DM                      dm;
        PetscSection            section;
        PetscBool               dmInterp = PETSC_TRUE;
        IS                      bcPointsIS;
        PetscInt                dim = 2, numFields, numBC;
        PetscScalar             lx = 1.0, ly = 2.0, phi = 0.2;
        PetscInt                numComp[1], numDOF[3], bcField[1];
        Vec			solVecLocal, solVecGlobal;
        PetscViewer             viewer, vtkviewer;

        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
        comm = PETSC_COMM_WORLD;

        ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
        ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, dmInterp, &dm);
        //        ierr  = DMPlexCreateFromFile(comm, "2Dtri3.exo", dmInterp, &dm);CHKERRQ(ierr);

        numFields = 1;
        numComp[0] = 1;
        for (PetscInt k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
        numDOF[0] = 1;
        numBC = 1;
        bcField[0] = 0;
        // Please note that bcField stays uninitialized because numBC = 0,
        // therefore having a trash value. This is probably handled internally
        // within DMPlexCreateSection but idk how exactly.

        ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
        ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
        ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
        ierr = DMSetSection(dm, section);CHKERRQ(ierr);

        //        ierr = StretchArray2D(dm, lx, ly);CHKERRQ(ierr);
        //ierr = ShearArray2D(dm, PETSC_PI/3);CHKERRQ(ierr);
        //ierr = SkewArray2D(dm, PETSC_PI/3);CHKERRQ(ierr);
        //ierr = LargeAngleDeformArray2D(dm, phi);CHKERRQ(ierr);
        //ierr = SmallAngleDeformArray2D(dm, phi);CHKERRQ(ierr);


        ierr = Stretch2DJacobian(dm, lx, ly);

        ierr = DMCreateGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
        //        ierr = VecSet(solVecGlobal, 0);CHKERRQ(ierr);
        ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
        //ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
        //ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
        //ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);

        ierr = PetscViewerCreate(comm, &vtkviewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(vtkviewer,PETSCVIEWERVTK);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(vtkviewer, "deformedmesh.vtk");CHKERRQ(ierr);
        ierr = VecView(solVecGlobal, vtkviewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&vtkviewer);CHKERRQ(ierr);

        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
