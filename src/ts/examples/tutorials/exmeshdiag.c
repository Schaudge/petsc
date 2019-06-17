static char help[] = "Diagnose mesh problems";

# include <petscdmplex.h>
# include <petscviewer.h>
# include <petscmath.h>

# define PETSCVIEWERASCII        "ascii"
# define PETSCVIEWERVTK          "vtk"
# define MATAIJ             	 "aij"
# define CHAR_LEN		 5

/*	2D Array Routines	*/
PetscErrorCode StretchArray2D(DM dm, PetscScalar lx, PetscScalar ly, char *deformId)
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
	strcat(deformId,"STR");
	return ierr;
}

PetscErrorCode ShearArray2D(DM dm, PetscScalar theta, char *deformId)
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
	strcat(deformId, "SHR_theta_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", theta);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) theta);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode SkewArray2D(DM dm, PetscScalar omega, char *deformId)
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
	strcat(deformId, "SKW_omega_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", omega);
	snprintf(phidecbuf, sizeof(phidecbuf),"%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) omega);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode LargeAngleDeformArray2D(DM dm, PetscScalar phi, char *deformId)
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

	/*	String formatting for file name	*/
	strcat(deformId, "LAD_phi_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", phi);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) phi);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

PetscErrorCode SmallAngleDeformArray2D(DM dm, PetscScalar phi, char *deformId)
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

	/*	String formatting for file name	*/
	strcat(deformId, "SAD_phi_");
	snprintf(tempbuf, sizeof(tempbuf), "%f", phi);
	snprintf(phidecbuf, sizeof(phidecbuf), "%s", strchr(tempbuf, '.')+1);
	snprintf(phiintbuf, sizeof(phiintbuf), "%d_", (int) phi);
	strcat(deformId, phiintbuf);
	strcat(deformId, phidecbuf);
	return ierr;
}

/*	2D Jacobians	*/
PetscErrorCode Stretch2DJacobian(DM dm, PetscScalar lx, PetscScalar ly, Mat *Jac)
{
        PetscErrorCode	ierr;
        DM		coordDM;
        PetscInt	pStart, pEnd;

        ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
        ierr = DMCreateMatrix(coordDM, Jac);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (PetscInt i = 0; i < (pEnd-pStart); i++) {
                PetscInt	II[2], J[2];
                PetscScalar	V[4];

                II[0] = 2*i;
                II[1] = 2*i+1;
                J[0] = 2*i;
                J[1] = 2*i+1;
                V[0] = lx;
                V[1] = 0;
                V[2] = 0;
                V[3] = ly;

                ierr = MatSetValuesLocal(*Jac, 2, II, 2, J, V, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode Shear2DJacobian(DM dm, PetscScalar theta, Mat *Jac)
{
        PetscErrorCode	ierr;
        DM		coordDM;
        PetscInt	pStart, pEnd;

        ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
        ierr = DMCreateMatrix(coordDM, Jac);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (PetscInt i = 0; i < (pEnd-pStart); i++) {
                PetscInt	II[2], J[2];
                PetscScalar	V[4];

                II[0] = 2*i;
                II[1] = 2*i+1;
                J[0] = 2*i;
                J[1] = 2*i+1;
                V[0] = 1;
                V[1] = PetscSinReal(theta);
                V[2] = 0;
                V[3] = PetscCosReal(theta);

                ierr = MatSetValuesLocal(*Jac, 2, II, 2, J, V, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode Skew2DJacobian(DM dm, PetscScalar omega, Mat *Jac)
{
        PetscErrorCode	ierr;
        DM		coordDM;
        PetscInt	pStart, pEnd;

        ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
        ierr = DMCreateMatrix(coordDM, Jac);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).

x         */

        for (PetscInt i = 0; i < (pEnd-pStart); i++) {
                PetscInt	II[2], J[2];
                PetscScalar	V[4];

                II[0] = 2*i;
                II[1] = 2*i+1;
                J[0] = 2*i;
                J[1] = 2*i+1;
                V[0] = PetscCosReal(omega);
                V[1] = 0;
                V[2] = PetscSinReal(omega);
                V[3] = 1;

                ierr = MatSetValuesLocal(*Jac, 2, II, 2, J, V, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode LargeAngle2DJacobian(DM dm, PetscScalar phi, Mat *Jac)
{
        PetscErrorCode	ierr;
        DM		coordDM;
        PetscScalar	*coordArray;
        PetscInt	pStart, pEnd;
        Vec		coords;

        if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }
	ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
	ierr = DMCreateMatrix(coordDM, Jac);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
	ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (PetscInt i = 0; i < (pEnd-pStart); i++) {
                PetscInt	II[2], J[2];
                PetscScalar	V[4];

                II[0] = 2*i;
                II[1] = 2*i+1;
                J[0] = 2*i;
                J[1] = 2*i+1;
                V[0] = 1-(phi*coordArray[2*i+1])+(phi/2);
                V[1] = (phi/2)-(phi*coordArray[2*i]);
                V[2] = -(phi/2)+(phi*coordArray[2*i+1]);
                V[3] = 1+(phi*coordArray[2*i])-(phi/2);

                ierr = MatSetValuesLocal(*Jac, 2, II, 2, J, V, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode SmallAngle2DJacobian(DM dm, PetscScalar phi, Mat *Jac)
{
        PetscErrorCode	ierr;
        DM		coordDM;
        PetscScalar	*coordArray;
        PetscInt	pStart, pEnd;
        Vec		coords;

        if ((phi < 0) || (phi > 1)) {
                MPI_Comm comm;
                ierr = PetscObjectGetComm((PetscObject) dm, &comm);
                SETERRQ(comm, 1, "Phi must be between [0,1]");
        }

        ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
        ierr = DMCreateMatrix(coordDM, Jac);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(dm, 0, &pStart, &pEnd);CHKERRQ(ierr);
        ierr = DMGetCoordinatesLocal(dm, &coords);CHKERRQ(ierr);
        ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);

        /*	 HOW JACOBIANS WORK IN PETSC: When you call DMCreateMatrix it pulls an
         already constructed but zeroed out jacobian matrix for you to insert
         values. Since our jacobian concerns 2 functions and 2 vars it is a 2x2 per
         VERTEX. So, in order to insert we must construct a 2x2 for every vertex, then
         pass that in as a flattened 1D array (V), and tell petsc which rows(II) and
         cols(J) to insert it into the big global jacobian(Jac).
         */

        for (PetscInt i = 0; i < (pEnd-pStart); i++) {
                PetscInt	II[2], J[2];
                PetscScalar	V[4];

                II[0] = 2*i;
                II[1] = 2*i+1;
                J[0] = 2*i;
                J[1] = 2*i+1;
                V[0] = 1;
                V[1] = 0;
                V[2] = phi-(2*phi*coordArray[2*i+1]);
                V[3] = 1-(2*phi*coordArray[2*i])+phi;

                ierr = MatSetValuesLocal(*Jac, 2, II, 2, J, V, INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
        ierr = MatAssemblyBegin(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        return ierr;
}

PetscErrorCode VTKPlotter(DM dm, char *deformId)
{
	PetscErrorCode		ierr;
	MPI_Comm		comm;
	PetscViewer		vtkviewer;
	Vec			outVecGlobal;

	strcat(deformId, ".vtk");

	ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
	ierr = DMGetGlobalVector(dm, &outVecGlobal);CHKERRQ(ierr);CHKERRQ(ierr);

	ierr = PetscViewerCreate(comm, &vtkviewer);CHKERRQ(ierr);
        ierr = PetscViewerSetType(vtkviewer,PETSCVIEWERVTK);CHKERRQ(ierr);
        ierr = PetscViewerFileSetName(vtkviewer, deformId);CHKERRQ(ierr);
        ierr = VecView(outVecGlobal, vtkviewer);CHKERRQ(ierr);

        ierr = PetscViewerDestroy(&vtkviewer);CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(dm, &outVecGlobal);CHKERRQ(ierr);
	return ierr;
}

int main(int argc, char **argv)
{
        PetscErrorCode          ierr;
        MPI_Comm                comm;
        DM                      dm, dmDist;
        PetscSection            section;
        PetscBool               dmInterp = PETSC_TRUE;
        IS                      bcPointsIS;
        PetscInt                dim = 2, numFields, numBC;
        PetscScalar             lx = 1.0, ly = 2.0, phi = 0.2;
        PetscInt                numComp[1], numDOF[3], bcField[1];
        Mat			Jac;

        ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
	comm = PETSC_COMM_WORLD;

	for (PetscInt i = 0; i < 9; i++){
		char		deformId[PETSC_MAX_PATH_LEN]="";
		/*	Convert deg to rad	*/
		PetscScalar 	dynamic_theta = 10*i*PETSC_PI/180.0;

		ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, dmInterp, &dm);CHKERRQ(ierr);
		//        ierr  = DMPlexCreateFromFile(comm, "2Dtri3.exo", dmInterp, &dm);CHKERRQ(ierr);

		ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
		if (dmDist) {ierr = DMDestroy(&dm);CHKERRQ(ierr); dm = dmDist;}

		numFields = 1;
		numComp[0] = 1;
		for (PetscInt k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
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

		//	  ierr = StretchArray2D(dm, lx, ly, deformId);CHKERRQ(ierr);
		//        ierr = ShearArray2D(dm, PETSC_PI/3, deformId);CHKERRQ(ierr);
		ierr = SkewArray2D(dm, dynamic_theta, deformId);CHKERRQ(ierr);
		//        ierr = LargeAngleDeformArray2D(dm, phi, deformId);CHKERRQ(ierr);
		//        ierr = SmallAngleDeformArray2D(dm, phi, deformId);CHKERRQ(ierr);
		printf("%s\n",deformId);
		//------------------------------------------------------------------
		/* Word to the wise: Don't try and combine the jacobian calls as they are
		   only meant to be called alone (i.e. every jacobian is unique to its
		   transform). I have left them out of the array modification routines
		   because those can be combined!!
		*/
		//------------------------------------------------------------------

		//        ierr = Stretch2DJacobian(dm, lx, ly, &Jac);CHKERRQ(ierr);
		//        ierr = Shear2DJacobian(dm, PETSC_PI/3, &Jac);CHKERRQ(ierr);
		//        ierr = Skew2DJacobian(dm, PETSC_PI/3, &Jac);CHKERRQ(ierr);
		//        ierr = LargeAngle2DJacobian(dm, phi, &Jac);CHKERRQ(ierr);
		//        ierr = SmallAngle2DJacobian(dm, phi, &Jac);CHKERRQ(ierr);

		ierr = VTKPlotter(dm, deformId);CHKERRQ(ierr);
	}
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        ierr = PetscFinalize();CHKERRQ(ierr);
        return ierr;
}
