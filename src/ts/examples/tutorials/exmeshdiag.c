static char help[] = "Diagnose mesh problems";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscmath.h>

# define PETSCVIEWERASCII        "ascii"
# define PETSCVIEWERVTK          "vtk"
# define CHAR_LEN		 5

/*	2D Array Routines	*/
PetscErrorCode StretchArray2D(DM dm, PetscScalar lx, PetscScalar ly, char *deformId, PetscInt *deformBoolArray)
{
        /*
         Analytical Jacobian:
         J = lx*ly
        */

        PetscErrorCode		ierr;
        PetscInt		i, nCoords;
        Vec			coordsLocal;
        PetscScalar		*coordArray;
	char			lintbuf[CHAR_LEN], ldecbuf[CHAR_LEN], tempbuf[CHAR_LEN];

	deformBoolArray[0] = 1;
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
	strcat(deformId,"STR_LX_");
	/*	lx	*/
	snprintf(tempbuf, sizeof(tempbuf), "%f", lx);
	snprintf(ldecbuf, sizeof(ldecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(lintbuf, sizeof(lintbuf), "%d_", (int) lx);
	strcat(deformId, lintbuf);
	strcat(deformId, ldecbuf);
	/*	ly	*/
	strcat(deformId,"_LY_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", ly);
	snprintf(ldecbuf, sizeof(ldecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(lintbuf, sizeof(lintbuf), "%d_", (int) ly);
	strcat(deformId, lintbuf);
	strcat(deformId, ldecbuf);
	return ierr;
}

PetscErrorCode ShearArray2D(DM dm, PetscScalar theta, char *deformId, PetscInt *deformBoolArray)
{
        /*
         Analytical Jacobian:
         J = cos(theta
        */


        PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsLocal;
        PetscScalar             *coordArray;
	char			phiintbuf[CHAR_LEN], phidecbuf[CHAR_LEN], tempbuf[CHAR_LEN];

	deformBoolArray[1] = 1;
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

	/*	String formatting for file name	*/
	strcat(deformId, "SHR_THTA_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", theta);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) theta);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode SkewArray2D(DM dm, PetscScalar omega, char *deformId, PetscInt *deformBoolArray)
{
        /*
         Analytical Jacobian:
         J = cos(omega)
        */

	PetscErrorCode          ierr;
        PetscInt                i, nCoords;
        Vec                     coordsLocal;
        PetscScalar             *coordArray;
	char			phiintbuf[CHAR_LEN]="", phidecbuf[CHAR_LEN]="", tempbuf[2*CHAR_LEN]="";

	deformBoolArray[2] = 1;
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

	/*	String formatting for file name	*/
	strcat(deformId, "SKW_OGA_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", omega);
	snprintf(phidecbuf, sizeof(phidecbuf),"%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) omega);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode LargeAngleDeformArray2D(DM dm, PetscScalar phi, char *deformId, PetscInt *deformBoolArray)
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
	char			phiintbuf[CHAR_LEN], phidecbuf[CHAR_LEN], tempbuf[CHAR_LEN];

        if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }
	deformBoolArray[3] = 1;
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

	/*	String formatting for file name	*/
	strcat(deformId, "LAD_Phi_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", phi);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) phi);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode SmallAngleDeformArray2D(DM dm, PetscScalar phi, char *deformId, PetscInt *deformBoolArray)
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
	char			phiintbuf[CHAR_LEN], phidecbuf[CHAR_LEN], tempbuf[CHAR_LEN];

	if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }
	deformBoolArray[4] = 1;
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

	/*	String formatting for file name	*/
	strcat(deformId, "SAD_Phi_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", phi);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) phi);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

/*	2D Jacobians	*/
PetscErrorCode Stretch2DJacobian(PetscInt xdim, PetscInt ydim, PetscScalar lx, PetscScalar ly, Mat Jac, Mat detJac)
{
        PetscErrorCode	ierr;
        PetscInt	row, col;
	PetscInt	II[2], J[2], Idet[1], Jdet[1];
	PetscScalar	V[4], detV[1];


        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (row = 0; row < ydim; row++) {
		II[0] = 2*row;
		II[1] = 2*row+1;
		Idet[0] = row;
		for (col = 0; col < xdim; col++) {
			J[0] = 2*col;
			J[1] = 2*col+1;
			V[0] = lx;
			V[1] = 0;
			V[2] = 0;
			V[3] = ly;

			Jdet[0] = col;
			detV[0] = lx*ly;

			ierr = MatSetValuesLocal(detJac, 1, Idet, 1, Jdet, detV, ADD_VALUES);CHKERRQ(ierr);
			ierr = MatSetValuesLocal(Jac, 2, II, 2, J, V, ADD_VALUES);CHKERRQ(ierr);
		}
        }
	ierr = MatAssemblyBegin(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode Shear2DJacobian(PetscInt xdim, PetscInt ydim, PetscScalar theta, Mat Jac, Mat detJac)
{
        PetscErrorCode	ierr;
        PetscInt	row, col;
	PetscInt	II[2], J[2], Idet[1], Jdet[1];
	PetscScalar	V[4], detV[1];


        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (row = 0; row < ydim; row++) {
		II[0] = 2*row;
		II[1] = 2*row+1;
		Idet[0] = row;
		for (col = 0; col < xdim; col++) {
			J[0] = 2*col;
			J[1] = 2*col+1;
			V[0] = 1;
			V[1] = PetscSinReal(theta);
			V[2] = 0;
			V[3] = PetscCosReal(theta);

			Jdet[0] = col;
			detV[0] = PetscCosReal(theta);

			ierr = MatSetValuesLocal(detJac, 1, Idet, 1, Jdet, detV, ADD_VALUES);CHKERRQ(ierr);
			ierr = MatSetValuesLocal(Jac, 2, II, 2, J, V, ADD_VALUES);CHKERRQ(ierr);
		}
        }
	ierr = MatAssemblyBegin(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode Skew2DJacobian(PetscInt xdim, PetscInt ydim, PetscScalar omega, Mat Jac, Mat detJac)
{
	PetscErrorCode	ierr;
        PetscInt	row, col;
	PetscInt	II[2], J[2], Idet[1], Jdet[1];
	PetscScalar	V[4], detV[1];


        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (row = 0; row < ydim; row++) {
		II[0] = 2*row;
		II[1] = 2*row+1;
		Idet[0] = row;
		for (col = 0; col < xdim; col++) {
			J[0] = 2*col;
			J[1] = 2*col+1;
			V[0] = PetscCosReal(omega);
			V[1] = 0;
			V[2] = PetscSinReal(omega);
			V[3] = 1;

			Jdet[0] = col;
			detV[0] = PetscCosReal(omega);

			ierr = MatSetValuesLocal(detJac, 1, Idet, 1, Jdet, detV, ADD_VALUES);CHKERRQ(ierr);
			ierr = MatSetValuesLocal(Jac, 2, II, 2, J, V, ADD_VALUES);CHKERRQ(ierr);
		}
        }
	ierr = MatAssemblyBegin(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode LargeAngle2DJacobian(DM dm, PetscInt xdim, PetscInt ydim, PetscScalar phi, Mat Jac, Mat detJac)
{
	PetscErrorCode	ierr;
	Vec		coords;
        PetscInt	coordi = 0, row, col;
	PetscInt	II[2], J[2], Idet[1], Jdet[1];
	PetscScalar	*coordArray, V[4], detV[1];

	if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }

        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (row = 0; row < ydim; row++) {
		II[0] = 2*row;
		II[1] = 2*row+1;
		Idet[0] = row;
		for (col = 0; col < xdim; col++) {
			J[0] = 2*col;
			J[1] = 2*col+1;
			V[0] = 1-(phi*coordArray[2*coordi+1])+(phi/2);
			V[1] = (phi/2)-(phi*coordArray[2*coordi]);
			V[2] = -(phi/2)+(phi*coordArray[2*coordi+1]);
			V[3] = 1+(phi*coordArray[2*coordi])-(phi/2);

			Jdet[0] = col;
			detV[0] = 1+phi*(coordArray[2*coordi]-coordArray[2*coordi+1]);

			ierr = MatSetValuesLocal(detJac, 1, Idet, 1, Jdet, detV, ADD_VALUES);CHKERRQ(ierr);
			ierr = MatSetValuesLocal(Jac, 2, II, 2, J, V, ADD_VALUES);CHKERRQ(ierr);
			coordi++;
		}
        }
        ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode SmallAngle2DJacobian(DM dm, PetscInt xdim, PetscInt ydim, PetscScalar phi, Mat Jac, Mat detJac)
{
	PetscErrorCode	ierr;
	Vec		coords;
        PetscInt	coordi = 0, row, col;
	PetscInt	II[2], J[2], Idet[1], Jdet[1];
	PetscScalar	*coordArray, V[4], detV[1];

	if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }

        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (row = 0; row < ydim; row++) {
		II[0] = 2*row;
		II[1] = 2*row+1;
		Idet[0] = row;
		for (col = 0; col < xdim; col++) {
			J[0] = 2*col;
			J[1] = 2*col+1;
			V[0] = 1;
			V[1] = 0;
			V[2] = phi-(2*phi*coordArray[2*coordi+1]);
			V[3] = 1-(2*phi*coordArray[2*coordi])+phi;

			Jdet[0] = col;
			detV[0] = 1-phi*(2*coordArray[2*coordi]-1);

			ierr = MatSetValuesLocal(detJac, 1, Idet, 1, Jdet, detV, ADD_VALUES);CHKERRQ(ierr);
			ierr = MatSetValuesLocal(Jac, 2, II, 2, J, V, ADD_VALUES);CHKERRQ(ierr);
			coordi++;
		}
        }
        ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(detJac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jac, MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

/*	Mesh Diagnostics	*/
PetscErrorCode DoesMyMeshSuck(MPI_Comm comm, DM dm, Mat allMats[],  PetscScalar *AggregateMeshScore, Vec *perCellMeshScore)
{
	PetscErrorCode	ierr;
	Vec 		OrthQual;

	ierr = VecDuplicate(*perCellMeshScore, &OrthQual);CHKERRQ(ierr);
	ierr = VecSetUp(OrthQual);CHKERRQ(ierr);
	ierr = VecZeroEntries(OrthQual);CHKERRQ(ierr);

	ierr = VecZeroEntries(*perCellMeshScore);CHKERRQ(ierr);
	*AggregateMeshScore = 0.0;
	ierr = GetCellJacobian(dm, allMats[1], perCellMeshScore);CHKERRQ(ierr);
	ierr = OrthoganalQuality(comm, dm, &OrthQual);CHKERRQ(ierr);

	ierr = VecCopy(OrthQual, *perCellMeshScore);CHKERRQ(ierr);
	ierr = VecDestroy(&OrthQual);CHKERRQ(ierr);
	return ierr;
}

/*	Visualization and Output	*/
PetscErrorCode VTKPlotter(MPI_Comm comm, DM dm, char *deformId, Mat outMat, Vec CellScore, PetscInt visID)
{
	PetscErrorCode		ierr;
	PetscViewer		vtkviewer;
	DM			dmLocal;
	PetscSF         	sfPoint;
        PetscSection    	coordSection, sectionLocal;
        Vec             	coordinates;
        PetscInt        	sStart, sEnd, sIter;

	/*	Make Temp DM	*/
	ierr = DMClone(dm, &dmLocal);CHKERRQ(ierr);
	/*	Get and Set Coord Map	*/
	ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
        ierr = DMSetCoordinateSection(dmLocal, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
	/*	Setup Starforest connectivity	*/
	ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
        ierr = DMSetPointSF(dmLocal, sfPoint);CHKERRQ(ierr);
	/*	Populate Coords	*/
	ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dmLocal, coordinates);CHKERRQ(ierr);
	/*	Create fake field to use as placeholder for Jacobian values 	*/
	ierr = PetscSectionCreate(comm , &sectionLocal);CHKERRQ(ierr);
        ierr = DMPlexGetHeightStratum(dmLocal, 0, &sStart, &sEnd);CHKERRQ(ierr);
        ierr = PetscSectionSetChart(sectionLocal, sStart, sEnd);CHKERRQ(ierr);

	for (sIter = sStart; sIter < sEnd; sIter++) {
                /*      Loop over cells, allow for assigning value       */
                ierr = PetscSectionSetDof(sectionLocal, sIter, 1);CHKERRQ(ierr);
        }
        ierr = PetscSectionSetUp(sectionLocal);CHKERRQ(ierr);
        ierr = DMSetSection(dmLocal, sectionLocal);CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&sectionLocal);CHKERRQ(ierr);

	switch(visID) {
	case 0:
		{
			SETERRQ(comm, ierr, "visID 0 isn't supported rn, would visualize jacobian master matrix, but you probably don't want to see that anyways");
			break;
		}
	case 1:
		/*	Visualize Jacobian determinant per cell	*/
		{
		/*	Gets a vector linking to the cells	*/
			Vec	cellJacobian;

			ierr = DMCreateLocalVector(dmLocal, &cellJacobian);CHKERRQ(ierr);
			ierr = PetscObjectSetName((PetscObject)cellJacobian, "Jacobian_value_per_cell");CHKERRQ(ierr);
			ierr = GetCellJacobian(dmLocal, outMat, &cellJacobian);CHKERRQ(ierr);

			strcat(deformId, "_JAC.vtk");
			ierr = PetscViewerCreate(comm, &vtkviewer);CHKERRQ(ierr);
			ierr = PetscViewerSetType(vtkviewer,PETSCVIEWERVTK);CHKERRQ(ierr);
			ierr = PetscViewerFileSetName(vtkviewer, deformId);CHKERRQ(ierr);
			ierr = VecView(cellJacobian, vtkviewer);CHKERRQ(ierr);

			ierr = VecDestroy(&cellJacobian);CHKERRQ(ierr);
			break;
		}
	case 2:
		{
		/*	Gets a vector linking to the cells	*/
			Mat	condCopy;

			ierr = MatDuplicate(outMat, MAT_COPY_VALUES, &condCopy);CHKERRQ(ierr);
			ierr = PetscObjectSetName((PetscObject)condCopy, "Condition_number_of_Jacobian_per_Vertex");CHKERRQ(ierr);

			strcat(deformId, "_COND.vtk");
			ierr = PetscViewerCreate(comm, &vtkviewer);CHKERRQ(ierr);
			ierr = PetscViewerSetType(vtkviewer,PETSCVIEWERVTK);CHKERRQ(ierr);
			ierr = PetscViewerFileSetName(vtkviewer, deformId);CHKERRQ(ierr);
			//ierr = MatView(condCopy, vtkviewer);CHKERRQ(ierr);

			ierr = MatDestroy(&condCopy);CHKERRQ(ierr);
			break;
		}
	case 3:
		/*	Visualize orthogonal qualit per cell	*/
		{
			Vec	cellScoreTemp;

			ierr = DMCreateLocalVector(dmLocal, &cellScoreTemp);CHKERRQ(ierr);
			ierr = VecCopy(CellScore, cellScoreTemp);CHKERRQ(ierr);
			ierr = PetscObjectSetName((PetscObject)cellScoreTemp, "Aggregate_mesh_score");CHKERRQ(ierr);
			strcat(deformId, "_mshScore.vtk");
			ierr = PetscViewerCreate(comm, &vtkviewer);CHKERRQ(ierr);
			ierr = PetscViewerSetType(vtkviewer, PETSCVIEWERVTK);CHKERRQ(ierr);
			ierr = PetscViewerFileSetName(vtkviewer, deformId);CHKERRQ(ierr);
			ierr = VecView(cellScoreTemp, vtkviewer);CHKERRQ(ierr);

			ierr = VecDestroy(&cellScoreTemp);CHKERRQ(ierr);
			break;
		}
	default:
		SETERRQ1(comm, ierr, "You gave an invalid visID = %d", visID);
		break;
	}
	ierr = PetscViewerDestroy(&vtkviewer);CHKERRQ(ierr);
	ierr = DMDestroy(&dmLocal);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode GeneralInfo(MPI_Comm comm, PetscViewer genViewer)
{
	PetscErrorCode 	ierr;
	char		bar[15] = "--------------";
	const char 	*string;
	size_t 		size;


	ierr = PetscPrintf(comm, "%s General Info %s\n", bar, bar);CHKERRQ(ierr);
	ierr = PetscViewerStringGetStringRead(genViewer, &string, &size);CHKERRQ(ierr);
	ierr = PetscPrintf(comm, string);CHKERRQ(ierr);
	ierr = PetscPrintf(comm, "%s End General Info %s\n", bar + 2, bar + 2);CHKERRQ(ierr);
	return ierr;
}

/*	Geometry	*/
PetscErrorCode GetCellJacobian(DM dm,  Mat detJac, Vec *cellJacobian)
{
	PetscErrorCode	ierr;
	IS		cells;
	PetscScalar	*arrayLocal, *matarray;
	PetscInt	sStart, sEnd, sIter;
	const PetscInt	*cellids;

        ierr = DMPlexGetHeightStratum(dm, 0, &sStart, &sEnd);CHKERRQ(ierr);
	ierr = VecGetArray(*cellJacobian, &arrayLocal);CHKERRQ(ierr);

	ierr = DMGetStratumIS(dm, "depth", 2, &cells);CHKERRQ(ierr);
	ierr = ISGetIndices(cells, &cellids);CHKERRQ(ierr);

	ierr = MatDenseGetArray(detJac, &matarray);CHKERRQ(ierr);
	for (sIter = sStart; sIter < sEnd; sIter++) {
		PetscInt	numPointsPerCell;
		ierr = DMPlexGetConeSize(dm, cellids[sIter], &numPointsPerCell);CHKERRQ(ierr);
		PetscInt	points[numPointsPerCell];
		ierr = Cell2Coords(dm, cellids[sIter], &points);CHKERRQ(ierr);
		ierr = CellAverageFromPoints(points, sEnd, numPointsPerCell, matarray, &arrayLocal[sIter]);CHKERRQ(ierr);
	}
	ierr = MatDenseRestoreArray(detJac, &matarray);CHKERRQ(ierr);
	ierr = VecRestoreArray(*cellJacobian, &arrayLocal);CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode Cell2Coords(DM dm, PetscInt cellId, PetscInt *points)
{
	PetscErrorCode	ierr;
	PetscInt	zeroiter, edgeIter, numConnEdges;
	const PetscInt	*connEdges;

	ierr = DMPlexGetCone(dm, cellId, &connEdges);CHKERRQ(ierr);
	ierr = DMPlexGetConeSize(dm, cellId, &numConnEdges);CHKERRQ(ierr);

	for (zeroiter = 0; zeroiter < numConnEdges; zeroiter++) {
		points[zeroiter] = -1;
	}
	for (edgeIter = 0;edgeIter < numConnEdges; edgeIter++) {
		PetscInt	pointIter, numConnPoints;
		const PetscInt	*connPoints;

		ierr = DMPlexGetCone(dm, connEdges[edgeIter], &connPoints);CHKERRQ(ierr);
		ierr = DMPlexGetConeSize(dm, connEdges[edgeIter], &numConnPoints);CHKERRQ(ierr);
		for (pointIter = 0; pointIter < numConnPoints; pointIter++) {
			PetscBool	inArray = PETSC_FALSE;

			valueInArray(connPoints[pointIter], points, numConnEdges, &inArray);
			if (!inArray) {
				points[edgeIter+pointIter] = connPoints[pointIter];
			}
		}
	}
	return ierr;
}

PetscErrorCode CentroidToFace(DM dm, const PetscInt cellid, PetscInt nPointsPerCell, Vec *cent2faces, Vec *faceNormVec)
{
	PetscErrorCode	ierr;
	PetscSection 	cSection;
	Vec 		cellCoord;
	const PetscInt	*faces;
	PetscInt	p, offset, minOff;
	PetscInt	points[nPointsPerCell];
	PetscScalar	xsum = 0.0, ysum = 0.0;
	PetscScalar	*cArray, *c2farr, centCoord[2], faceCent[2];

	ierr = DMGetCoordinatesLocal(dm, &cellCoord);CHKERRQ(ierr);
	ierr = DMGetCoordinateSection(dm, &cSection);CHKERRQ(ierr);
	ierr = VecGetArray(cellCoord, &cArray);CHKERRQ(ierr);
	ierr = VecGetArray(*cent2faces, &c2farr);CHKERRQ(ierr);
	ierr = Cell2Coords(dm, cellid, &points);CHKERRQ(ierr);
	for (p = 0; p < nPointsPerCell; p++) {
		ierr = PetscSectionGetOffset(cSection, points[p], &offset);CHKERRQ(ierr);
		xsum += cArray[offset];
		ysum += cArray[offset + 1];
	}
	centCoord[0] = xsum/(PetscScalar)nPointsPerCell;
	centCoord[1] = ysum/(PetscScalar)nPointsPerCell;

	ierr = DMPlexGetCone(dm, cellid, &faces);CHKERRQ(ierr);
	ierr = PetscSectionGetOffsetRange(cSection, &minOff, NULL);CHKERRQ(ierr);
	for (p = 0; p < nPointsPerCell; p++) {
		const PetscInt	face = faces[p];
		const PetscInt	*facePoints;
		PetscInt	i;
		xsum = 0.0; ysum = 0.0;
		ierr = DMPlexGetCone(dm, face, &facePoints);CHKERRQ(ierr);
		for (i = 0; i < 2; i++) {
			ierr = PetscSectionGetOffset(cSection, facePoints[i], &offset);CHKERRQ(ierr);
			xsum += cArray[offset-minOff];
			ysum += cArray[offset-minOff + 1];
		}
		faceCent[0] = xsum/2.0;
		faceCent[1] = ysum/2.0;
		c2farr[2*p] = faceCent[0]-centCoord[0];
		c2farr[2*p + 1] = faceCent[1]-centCoord[1];
		ierr = FaceNormPerCell(dm, cSection, facePoints, p, faceNormVec);CHKERRQ(ierr);
	}
	ierr = VecRestoreArray(cellCoord, &cArray);CHKERRQ(ierr);
	ierr = VecRestoreArray(*cent2faces, &c2farr);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode FaceNormPerCell(DM dm, PetscSection cSection, const PetscInt faceid[], PetscInt idx,  Vec *faceNormVec)
{
	PetscErrorCode	ierr;
	Vec		coords;
	PetscInt	offset0, offset1, minOff;
	PetscScalar	dx = 0.0, dy = 0.0;
	PetscScalar	*cArray, *fArray;

	ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
	ierr = VecGetArray(coords, &cArray);CHKERRQ(ierr);
	ierr = VecGetArray(*faceNormVec, &fArray);CHKERRQ(ierr);
	ierr = PetscSectionGetOffsetRange(cSection, &minOff, NULL);CHKERRQ(ierr);
	ierr = PetscSectionGetOffset(cSection, faceid[1], &offset1);CHKERRQ(ierr);
	ierr = PetscSectionGetOffset(cSection, faceid[0], &offset0);CHKERRQ(ierr);

	dx = cArray[offset1-minOff]-cArray[offset0-minOff];
	dy = cArray[offset1-minOff+1]-cArray[offset0-minOff+1];
	fArray[2*idx] = -dy;
	fArray[2*idx + 1] = dx;

	ierr = VecRestoreArray(coords, &cArray);CHKERRQ(ierr);
	ierr = VecRestoreArray(*faceNormVec, &fArray);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode OrthoganalQuality(MPI_Comm comm, DM dm, Vec *OrthQual)
{
	PetscErrorCode	ierr;
	IS		cellIS, subAlloc;
	const PetscInt	*cells;
	PetscInt	cStart, cEnd, cellIter, nPointsPerCell, i;
	Vec		temp, cent2faces, faceNormVec;

	ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
	ierr = DMGetStratumIS(dm, "depth", 2, &cellIS);CHKERRQ(ierr);
	ierr = ISGetIndices(cellIS, &cells);CHKERRQ(ierr);
	ierr = DMPlexGetConeSize(dm, cells[0], &nPointsPerCell);CHKERRQ(ierr);

	ierr = VecCreate(comm, &temp);CHKERRQ(ierr);
	ierr = VecSetSizes(temp, PETSC_DECIDE, nPointsPerCell);CHKERRQ(ierr);
	ierr = VecSetUp(temp);CHKERRQ(ierr);
	ierr = VecZeroEntries(temp);CHKERRQ(ierr);

	ierr = VecCreate(comm, &cent2faces);CHKERRQ(ierr);
	ierr = VecSetSizes(cent2faces, PETSC_DECIDE, 2*nPointsPerCell);CHKERRQ(ierr);
	ierr = VecSetBlockSize(cent2faces, 2);CHKERRQ(ierr);
	ierr = VecSetUp(cent2faces);CHKERRQ(ierr);
	ierr = VecZeroEntries(cent2faces);CHKERRQ(ierr);
	ierr = VecDuplicate(cent2faces, &faceNormVec);CHKERRQ(ierr);
	ierr = VecCopy(cent2faces, faceNormVec);CHKERRQ(ierr);
	ierr = VecSetUp(faceNormVec);CHKERRQ(ierr);

	for (cellIter = cStart; cellIter < cEnd; cellIter++) {
		const PetscInt	cell = cells[cellIter];
		PetscScalar	OrthQualPerFace = 0.0, OrthQualPerCell = 0.0, Anorm, Fnorm, DotProd= 0.0;
		ierr = VecZeroEntries(cent2faces);CHKERRQ(ierr);
		ierr = VecZeroEntries(faceNormVec);CHKERRQ(ierr);
		ierr = CentroidToFace(dm, cell, nPointsPerCell, &cent2faces, &faceNormVec);CHKERRQ(ierr);
		for (i = 0; i < nPointsPerCell; i++) {
			PetscInt	*idx;
			size_t		size=2;
			Vec  		subVecCent, subVecFace;

			ierr = PetscMalloc1(size, &idx);CHKERRQ(ierr);
			idx[0] = 2*i; idx[1] = 2*i+1;
			ierr = ISCreateGeneral(PETSC_COMM_WORLD, 2, idx, PETSC_COPY_VALUES, &subAlloc);CHKERRQ(ierr);
			ierr = PetscFree(idx);CHKERRQ(ierr);
			ierr = VecGetSubVector(cent2faces, subAlloc, &subVecCent);CHKERRQ(ierr);
			ierr = VecGetSubVector(faceNormVec, subAlloc, &subVecFace);CHKERRQ(ierr);
			ierr = VecNorm(subVecCent, NORM_2, &Fnorm);CHKERRQ(ierr);
			ierr = VecNorm(subVecFace, NORM_2, &Anorm);CHKERRQ(ierr);
			ierr = VecDot(subVecCent, subVecFace, &DotProd);CHKERRQ(ierr);
			OrthQualPerFace = DotProd/(Fnorm*Anorm);
			ierr = VecSetValue(temp, i, OrthQualPerFace, INSERT_VALUES);CHKERRQ(ierr);
			ierr = VecAssemblyBegin(temp);CHKERRQ(ierr);
			ierr = VecAssemblyEnd(temp);CHKERRQ(ierr);
			ierr = VecRestoreSubVector(cent2faces, subAlloc, &subVecCent);CHKERRQ(ierr);
			ierr = VecRestoreSubVector(faceNormVec, subAlloc, &subVecFace);CHKERRQ(ierr);
			ierr = ISDestroy(&subAlloc);CHKERRQ(ierr);
		}
		ierr = VecAbs(temp);CHKERRQ(ierr);
		ierr = VecMin(temp, NULL, &OrthQualPerCell);CHKERRQ(ierr);
		ierr = VecSetValue(*OrthQual, cellIter, OrthQualPerCell, INSERT_VALUES);CHKERRQ(ierr);
		ierr = VecAssemblyBegin(*OrthQual);CHKERRQ(ierr);
		ierr = VecAssemblyEnd(*OrthQual);CHKERRQ(ierr);
	}
	ierr = ISRestoreIndices(cellIS, &cells);CHKERRQ(ierr);
	ierr = VecDestroy(&temp);CHKERRQ(ierr);
	ierr = VecDestroy(&faceNormVec);CHKERRQ(ierr);
	ierr = VecDestroy(&cent2faces);CHKERRQ(ierr);

	return ierr;
}

/*	General Helper	*/
int CellAverageFromPoints(PetscInt *points, PetscInt numCells, PetscInt numPointsPerCell,  PetscScalar *matarray, PetscScalar *arrayLocal)
{
	PetscInt	i;
	PetscScalar	total = 0.0, avg = 0.0;

	for (i = 0; i < numPointsPerCell; i++){
		total += matarray[points[i]-numCells];
	}
	avg = total/(PetscScalar)numPointsPerCell;
	*arrayLocal += avg;
	return 0;
}

int valueInArray(const PetscInt val, PetscInt *arr, PetscInt sizeOfArr, PetscBool *inArray)
{
	int i;
	for(i = 0; i < sizeOfArr; i++) {
		if(arr[i] == val) {
			*inArray = PETSC_TRUE;
			return 1;
		}
	}
	*inArray = PETSC_FALSE;
	return 0;
}

PetscErrorCode ConditionNumber2x2(MPI_Comm comm, Mat Jac, Mat condJac)
{
	PetscErrorCode	ierr;
	IS		subMatISX, subMatISY;
	PetscInt	row, col, numSubMatsX, numSubMatsY, subMatIterX, subMatIterY;
	PetscInt	idx[2], jdx[2];
	PetscScalar	regNorm, invNorm, condNumber;
	Mat 		Imat, subMat, inverseMat;

	ierr = MatGetSize(Jac, &row, &col);CHKERRQ(ierr);
	numSubMatsX = col/2;
	numSubMatsY = row/2;

	ierr = MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2, 2, NULL, &Imat);CHKERRQ(ierr);
	ierr = MatDuplicate(Imat, MAT_DO_NOT_COPY_VALUES, &inverseMat);CHKERRQ(ierr);
	ierr = MatSetValue(Imat, 0, 0, 1.0, INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatSetValue(Imat, 1, 1, 1.0, INSERT_VALUES);CHKERRQ(ierr);
	ierr = MatSetUp(Imat);CHKERRQ(ierr);
	ierr = MatSetUp(inverseMat);CHKERRQ(ierr);
	for (subMatIterX = 0; subMatIterX < numSubMatsX; subMatIterX++) {
		idx[0] = 2*subMatIterX;
		idx[1] = (2*subMatIterX)+1;
		ierr = ISCreateGeneral(comm, 2, idx, PETSC_COPY_VALUES, &subMatISX);CHKERRQ(ierr);

		for (subMatIterY = 0; subMatIterY < numSubMatsY; subMatIterY++) {
			jdx[0] = 2*subMatIterY;
			jdx[1] = (2*subMatIterY)+1;

			ierr = ISCreateGeneral(comm, 2, jdx, PETSC_COPY_VALUES, &subMatISY);CHKERRQ(ierr);
			ierr = MatCreateSubMatrix(Jac, subMatISX, subMatISY, MAT_INITIAL_MATRIX, &subMat);CHKERRQ(ierr);
			ierr = MatNorm(subMat, NORM_1, &regNorm);CHKERRQ(ierr);

			ierr = Invert2x2(comm, subMat, inverseMat);CHKERRQ(ierr);
			ierr = MatNorm(inverseMat, NORM_1, &invNorm);CHKERRQ(ierr);
			condNumber = regNorm/invNorm;

			ierr = MatSetValue(condJac, subMatIterX, subMatIterY, condNumber, INSERT_VALUES);CHKERRQ(ierr);

			ierr = MatDestroy(&subMat);CHKERRQ(ierr);
			ierr = ISDestroy(&subMatISY);CHKERRQ(ierr);
		}
		ierr = ISDestroy(&subMatISX);CHKERRQ(ierr);
	}
	ierr = MatDestroy(&Imat);CHKERRQ(ierr);
	ierr = MatDestroy(&inverseMat);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode Invert2x2(MPI_Comm comm, Mat regMat, Mat inMat)
{
	PetscErrorCode		ierr;
	PetscInt		m, n;
	const PetscScalar	*regMatArray;
	PetscScalar		*inMatArray;
	PetscScalar		aa, bb, cc, dd, det;

	ierr = MatGetSize(regMat, &m, &n);CHKERRQ(ierr);
	if ((m != 2) || (n != 2)) {
		SETERRQ2(comm, ierr, "Matrix is not a 2x2, is a %dx%d", m, n);CHKERRQ(ierr);
	} else {
		ierr = MatDenseGetArrayRead(regMat, &regMatArray);CHKERRQ(ierr);
		ierr = MatDenseGetArray(inMat, &inMatArray);CHKERRQ(ierr);
		aa = regMatArray[0]; bb = regMatArray[2]; cc = regMatArray[1]; dd= regMatArray[3];
		det = (aa*dd)-(bb*cc);
		if (det != 0)  {
			inMatArray[0] = dd/det; inMatArray[1] = -cc/det; inMatArray[2] = -bb/det; inMatArray[3] = aa/det;
			ierr = MatDenseRestoreArrayRead(regMat, &regMatArray);CHKERRQ(ierr);
			ierr = MatDenseRestoreArray(inMat, &inMatArray);CHKERRQ(ierr);
		} else {
			// Matrix is singular, this shouldn't happen, you have somehow
			// managed to fuck this up
			SETERRQ(comm, ierr, "Can't invert a singular matrix my guy");
		}
	}
	return ierr;
}

int main(int argc, char **argv)
{
        PetscErrorCode          ierr;
        MPI_Comm                comm;
	PetscViewer		genViewer;
        DM                      dm, dmDist;
        PetscSection            section;
        PetscBool               dmInterp = PETSC_TRUE;
        IS                      bcPointsIS;
	ISLocalToGlobalMapping	ltogmap, ltogmapJac;
	Vec			perCellMeshScore;
        PetscInt                i, k, booli, dim = 2, xdim,  ydim, numFields, numBC, visID = 1, nCells;
        PetscScalar             lx = 1.0, ly = 1.0, phi = 0.2, AggregateMeshScore;
        PetscInt                deformBoolArray[5], faces[2], numComp[1], numDOF[3], bcField[1];
        Mat			Jac, detJac, condJac, allMats[4];
	char			genInfo[2048];

        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
	comm = PETSC_COMM_WORLD;
	ierr = PetscViewerStringOpen(comm, genInfo, sizeof(genInfo), &genViewer);CHKERRQ(ierr);

	for (i = 0; i < 4; i++){
		ierr = PetscViewerStringSPrintf(genViewer, "Run %d: ", i);CHKERRQ(ierr);
		for (booli = 0; booli < 5; booli++){
		/*	Array used to check which deform was used	*/
			deformBoolArray[booli] = 0;
		}
		char		deformId[PETSC_MAX_PATH_LEN]="";
		/*	Convert deg to rad	*/
		PetscScalar 	dynamic_theta = 10*i*PETSC_PI/180.0;

		faces[0] = 2;
		faces[1] = 2;
		xdim = faces[0] + 1;
		ydim = faces[1] + 1;
		ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL, NULL, NULL, dmInterp, &dm);CHKERRQ(ierr);

		ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
		if (dmDist) {ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist;}

		numFields = 1;
		numComp[0] = 1;
		for (k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
		numDOF[0] = 1;
		numBC = 1;
		bcField[0] = 0;

		// Please note that bcField stays uninitialized because numBC = 0, therefore
		// having a trash value. This is probably handled internally within
		// DMPlexCreateSection but idk how exactly.

		ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
		ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
		ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
		ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
		ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
		ierr = DMSetSection(dm, section);CHKERRQ(ierr);

		/*	Make all Mats	*/
		ierr = MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2*xdim, 2*ydim, NULL, &Jac);CHKERRQ(ierr);
		ierr = DMGetLocalToGlobalMapping(dm, &ltogmapJac);CHKERRQ(ierr);
		ierr = MatSetLocalToGlobalMapping(Jac, ltogmapJac, ltogmapJac);CHKERRQ(ierr);
		ierr = MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, xdim, ydim, NULL, &detJac);CHKERRQ(ierr);
		ierr = DMGetLocalToGlobalMapping(dm, &ltogmap);CHKERRQ(ierr);
		ierr = MatSetLocalToGlobalMapping(detJac, ltogmap, ltogmap);CHKERRQ(ierr);
		ierr = MatDuplicate(detJac,  MAT_DO_NOT_COPY_VALUES, &condJac);CHKERRQ(ierr);

		ierr = MatZeroEntries(Jac);CHKERRQ(ierr);
		ierr = MatZeroEntries(detJac);CHKERRQ(ierr);
		ierr = MatZeroEntries(condJac);CHKERRQ(ierr);

		/*	Deformations	*/
		ierr = StretchArray2D(dm, lx, ly, deformId, deformBoolArray);CHKERRQ(ierr);
		ierr = ShearArray2D(dm, dynamic_theta, deformId, deformBoolArray);CHKERRQ(ierr);
		ierr = SkewArray2D(dm, dynamic_theta, deformId, deformBoolArray);CHKERRQ(ierr);
		ierr = LargeAngleDeformArray2D(dm, phi, deformId, deformBoolArray);CHKERRQ(ierr);
		ierr = SmallAngleDeformArray2D(dm, phi, deformId, deformBoolArray);CHKERRQ(ierr);

		/*	Jacobian Generation	*/
		if (deformBoolArray[0]){ierr = Stretch2DJacobian(xdim, ydim, lx, ly, Jac, detJac);CHKERRQ(ierr);ierr = PetscViewerStringSPrintf(genViewer, "Stretch used ");CHKERRQ(ierr);}
		if (deformBoolArray[1]){ierr = Shear2DJacobian(xdim, ydim, PETSC_PI/3, Jac, detJac);CHKERRQ(ierr);ierr = PetscViewerStringSPrintf(genViewer, "Shear used ");CHKERRQ(ierr);}
		if (deformBoolArray[2]){ierr = Skew2DJacobian(xdim, ydim, PETSC_PI/3, Jac, detJac);CHKERRQ(ierr);ierr = PetscViewerStringSPrintf(genViewer, "Skew used ");CHKERRQ(ierr);}
		if (deformBoolArray[3]){ierr = LargeAngle2DJacobian(dm, xdim, ydim, phi, Jac, detJac);CHKERRQ(ierr);ierr = PetscViewerStringSPrintf(genViewer, "LA used ");CHKERRQ(ierr);}
		if (deformBoolArray[4]){ierr = SmallAngle2DJacobian(dm, xdim, ydim, phi, Jac, detJac);CHKERRQ(ierr);ierr = PetscViewerStringSPrintf(genViewer, "SA used ");CHKERRQ(ierr);}

		/*	Assemble EVERYTHING	*/
		ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyBegin(detJac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(detJac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = ConditionNumber2x2(comm, Jac, condJac);CHKERRQ(ierr);
		ierr = MatAssemblyBegin(condJac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(condJac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

		/*	Output	*/
		allMats[0] = Jac;
		allMats[1] = detJac;
		allMats[2] = condJac;
		allMats[3] = NULL;
		// Change visID above to display your desired output, so far only 1, 2 are
		// supported. allMats[3] is NULL so that I can plot the mesh score. Check
		// VTKPlotter to see what they plot.
		ierr = DMPlexGetDepthStratum(dm, 2, NULL, &nCells);CHKERRQ(ierr);
		ierr = VecCreate(comm, &perCellMeshScore);CHKERRQ(ierr);
		ierr = VecSetSizes(perCellMeshScore, PETSC_DECIDE, nCells);CHKERRQ(ierr);
		ierr = VecSetUp(perCellMeshScore);CHKERRQ(ierr);
		ierr = VecZeroEntries(perCellMeshScore);CHKERRQ(ierr);
		ierr = DoesMyMeshSuck(comm, dm, allMats,  &AggregateMeshScore, &perCellMeshScore);CHKERRQ(ierr);
		ierr = VTKPlotter(comm, dm, deformId, allMats[visID], perCellMeshScore, visID);CHKERRQ(ierr);
		ierr = PetscViewerStringSPrintf(genViewer, "\n->Wrote vtk to: %s\n", deformId);CHKERRQ(ierr);
		ierr = MatDestroy(&Jac);CHKERRQ(ierr);
		ierr = MatDestroy(&detJac);CHKERRQ(ierr);
		ierr = MatDestroy(&condJac);CHKERRQ(ierr);
		ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
		ierr = VecDestroy(&perCellMeshScore);CHKERRQ(ierr);
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
	}
	ierr = PetscViewerStringSPrintf(genViewer, "Total runs: %d\n", i);CHKERRQ(ierr);
	ierr = GeneralInfo(comm, genViewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);

        return ierr;
}

/* TODO:
-implement vec field from 2x2 jac, where 2 vecs are generated from [dx/dr dx/ds] and [dy/dr
dy/ds], maybe matlab
-look at gmsh mesh diagnostics
-look at ansys mesh diagnostics
-aspect ratio
*/
