static char help[] = "Test Unstructured Mesh Handling\n";

# include <petscdmplex.h>
# include <petscviewer.h>

# define PETSCVIEWERVTK          "vtk"
# define PETSCVIEWERASCII        "ascii"
# define VECSTANDARD    	 "standard"

/*	ADDITIONAL FUNCTIONS	*/
PetscErrorCode VTKPartitionVisualize(DM dm, DM *dmLocal, Vec *partition)
{
        MPI_Comm	DMcomm;
        PetscSF        	sfPoint;
	PetscSection   	coordSection;
	Vec            	coordinates;
	PetscSection   	sectionLocal;
	PetscScalar    	*arrayLocal;
	PetscInt       	sStart, sEnd, sIter;
	PetscMPIInt    	rank;
	PetscErrorCode 	ierr;


	/*	Make Temp DM	*/
	ierr = DMClone(dm, dmLocal);CHKERRQ(ierr);
	/*	Get and Set Coord Map	*/
	ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
	ierr = DMSetCoordinateSection(*dmLocal, PETSC_DETERMINE, coordSection);CHKERRQ(ierr);
	/*	Get and Set Neighbors	*/
	ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
	ierr = DMSetPointSF(*dmLocal, sfPoint);CHKERRQ(ierr);
	/*	Populate Coords		*/
	ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
	ierr = DMSetCoordinatesLocal(*dmLocal, coordinates);CHKERRQ(ierr);
	/*	Get Local Comm handle	*/
	ierr = PetscObjectGetComm((PetscObject) *dmLocal, &DMcomm);CHKERRQ(ierr);
	ierr = MPI_Comm_rank(DMcomm , &rank);CHKERRQ(ierr);
	/*	Setup the partition "field"	*/
	ierr = PetscSectionCreate(DMcomm , &sectionLocal);CHKERRQ(ierr);
	ierr = DMPlexGetHeightStratum(*dmLocal, 0, &sStart, &sEnd);CHKERRQ(ierr);
	ierr = PetscSectionSetChart(sectionLocal, sStart, sEnd);CHKERRQ(ierr);

	for (sIter = sStart; sIter < sEnd; ++sIter) {
		/*	Allow for assigning value	*/
		ierr = PetscSectionSetDof(sectionLocal, sIter, 1);CHKERRQ(ierr);
	}

	ierr = PetscSectionSetUp(sectionLocal);CHKERRQ(ierr);
	ierr = DMSetSection(*dmLocal, sectionLocal);CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&sectionLocal);CHKERRQ(ierr);
	ierr = DMCreateLocalVector(*dmLocal, partition);CHKERRQ(ierr);
	ierr = VecGetArray(*partition, &arrayLocal);CHKERRQ(ierr);
	ierr = PetscObjectSetName((PetscObject)*partition, "Partition_per_process");CHKERRQ(ierr);

	for (sIter = sStart; sIter < sEnd; ++sIter) {
		arrayLocal[sIter] = rank;
	}
	ierr = VecRestoreArray(*partition, &arrayLocal);CHKERRQ(ierr);
	return ierr;
}

PetscErrorCode ViewISInfo(MPI_Comm comm, DM dm)
{
	PetscViewer	viewer;
	DMLabel		label;
	IS 		labelIS;
	const char 	*labelName;
	PetscInt 	numLabels, l;
	PetscErrorCode 	ierr;
	char            tbar[10] = "----------";

	ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
        /*	query the number and name of labels	*/
        ierr = DMGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
       	ierr = PetscPrintf(comm, "Number of labels: %d\n", numLabels);CHKERRQ(ierr);
        for (l = 0; l < numLabels; ++l)
        {
        	ierr = DMGetLabelName(dm, l, &labelName);CHKERRQ(ierr);
        	ierr = PetscPrintf(comm, "Label %d: name: %s\n", l, labelName);CHKERRQ(ierr);
                ierr = PetscPrintf(comm, "IS of values\n");CHKERRQ(ierr);
                ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);
                ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
                ierr = ISView(labelIS, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
                ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
                ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
        }
        /*	Making sure that string literals work	*/
        ierr = PetscPrintf(comm,"\n\nCell Set label IS\n");CHKERRQ(ierr);
        ierr = DMGetLabel(dm, "Cell Sets", &label);CHKERRQ(ierr);
	/*	Specifically look for Cell Sets as these seem to be vertices	*/
        if (label)
        {
        	ierr = DMLabelGetValueIS(label, &labelIS);CHKERRQ(ierr);
                ierr = ISView(labelIS, viewer);CHKERRQ(ierr);
                ierr = ISDestroy(&labelIS);CHKERRQ(ierr);
        }
        ierr = PetscPrintf(comm,"%s End Label View %s\n", tbar, tbar);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        return ierr;
}

/* 	Main	*/
int main(int argc, char **argv)
{
	MPI_Comm		comm;
	PetscErrorCode		ierr;
	PetscLogStage 		stage;
	PetscLogEvent 		event;
	DM			dm, dmDist, dmInterp;
	IS			bcPointsIS;
	PetscSection		section;
	Vec			funcVecSin, funcVecCos, solVecLocal, solVecGlobal, coordinates;
	PetscBool		fileflg = PETSC_FALSE, dmInterped = PETSC_TRUE, dispFlag = PETSC_FALSE, isView = PETSC_FALSE,  VTKdisp = PETSC_FALSE, dmDisp = PETSC_FALSE, sectionDisp = PETSC_FALSE, arrayDisp = PETSC_FALSE, coordDisp = PETSC_FALSE;
	PetscInt		dim = 2, i, j, k, numFields, numBC, vecsize = 1000, nCoords, nVertex;
	PetscInt		numComp[3], numDOF[3], bcField[1];
        size_t                  namelen;
	PetscScalar 		dot, *coords, *array;
	PetscViewer		viewer;
	char			bar[18] = "------------------", filename[PETSC_MAX_PATH_LEN];

	ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
	comm = PETSC_COMM_WORLD;

	ierr = PetscOptionsBegin(comm, NULL, "Speedtest Options", "");CHKERRQ(ierr); {
		ierr = PetscOptionsBool("-vtkout", "Enable mesh distribution visualization", "", VTKdisp, &VTKdisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-disp", "Turn on all displays", "", dispFlag, &dispFlag, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-isview", "Turn on ISView for single threaded", "", isView, &isView, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-dmview", "Turn on DMView", "", dmDisp, &dmDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-secview","Turn on SectionView", "", sectionDisp, &sectionDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-arrview", "Turn on array display", "", arrayDisp, &arrayDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-coordview","Turn on coordinate display", "", coordDisp, &coordDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &fileflg); CHKERRQ(ierr);
	}
	ierr = PetscOptionsEnd();CHKERRQ(ierr);
	if (dispFlag) {isView = PETSC_TRUE; dmDisp = PETSC_TRUE; sectionDisp = PETSC_TRUE, arrayDisp = PETSC_TRUE; coordDisp = PETSC_TRUE;}

	ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
	ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);

        ierr = PetscStrlen(filename, &namelen);CHKERRQ(ierr);
        if (!namelen){
          	ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, NULL, NULL, NULL, NULL, dmInterped, &dm);CHKERRQ(ierr);
        } else {
          	ierr = DMPlexCreateFromFile(comm, filename, dmInterped, &dm);CHKERRQ(ierr);
        }

	ierr = PetscPrintf(comm,"%s Dm Alteration Info %s \n", bar, bar);CHKERRQ(ierr);
	ierr = DMPlexDistribute(dm, 0, NULL, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"Distributed dm\n");CHKERRQ(ierr);
		dm = dmDist;
	}else{
		ierr = PetscPrintf(comm,"No distributed dm\n");CHKERRQ(ierr);
		if (isView) {
                        ierr = PetscPrintf(comm, "%s Label View %s\n",bar, bar);CHKERRQ(ierr);
			ierr = ViewISInfo(comm, dm);CHKERRQ(ierr);
		}
	}
	if (!dmInterped) {
		ierr = DMPlexInterpolate(dm, &dmInterp);CHKERRQ(ierr);
		if (dmInterp) {
			ierr = DMDestroy(&dm);CHKERRQ(ierr);
			ierr = PetscPrintf(comm,"Interped dm again [UNUSUAL]\n");CHKERRQ(ierr);
			dm = dmInterp;
		}else{
			ierr = PetscPrintf(comm,"No interped dm [QUITE UNUSUAL]\n");CHKERRQ(ierr);
		}
	}else{
		ierr = PetscPrintf(comm,"Interped dm\n");CHKERRQ(ierr);
	}
	ierr = PetscPrintf(comm,"%s End Dm Alteration %s\n", bar, bar);CHKERRQ(ierr);

	/*	Set number of active fields	*/
	numFields = 1;
	/* 	Number of Field Components	*/
	numComp[0] = 1;
	/*	Init numDOF[field componentID] = Not Used	*/
	for (k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
	/*	numDOF[field componentID] = Used	*/
	numDOF[0] = 1;
	/*	numComp[componentID] = Used	*/
	numComp[0] = 1;
	/*	Init number of boundary conditions	*/
	numBC = 1;
	/*	bcField[boundary conditionID] = Dirichtlet Val	*/
	bcField[0] = 0;

	/*	Assign BC using IS of LOCAL boundaries	*/
        ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
	ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
	ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
	ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
	ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
	ierr = DMSetSection(dm, section);CHKERRQ(ierr);
	if (dmDisp) {
		ierr = PetscPrintf(comm,"%s DM View %s\n", bar, bar);CHKERRQ(ierr);
		ierr = DMView(dm, 0);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s End DM View %s\n", bar, bar);CHKERRQ(ierr);
	}

	/*	Perform Function on LOCAL array	*/
	ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
	ierr = VecGetLocalSize(solVecLocal, &nVertex);CHKERRQ(ierr);
	ierr = VecGetArray(solVecLocal, &array);CHKERRQ(ierr);

	/*	Create Vector for per process function evaluation	*/
	ierr = VecCreate(PETSC_COMM_SELF, &funcVecSin);CHKERRQ(ierr);
	ierr = VecSetType(funcVecSin, VECSTANDARD);CHKERRQ(ierr);
	ierr = VecSetSizes(funcVecSin, PETSC_DECIDE, vecsize);CHKERRQ(ierr);
	ierr = VecSetFromOptions(funcVecSin);CHKERRQ(ierr);
	ierr = VecDuplicate(funcVecSin, &funcVecCos);CHKERRQ(ierr);
	ierr = VecSet(funcVecSin, PetscSinReal(PETSC_PI));CHKERRQ(ierr);
	ierr = VecSet(funcVecCos, PetscCosReal(PETSC_PI));CHKERRQ(ierr);
	ierr = VecAssemblyBegin(funcVecSin);CHKERRQ(ierr);
  	ierr = VecAssemblyEnd(funcVecSin);CHKERRQ(ierr);
	ierr = VecAssemblyBegin(funcVecCos);CHKERRQ(ierr);
  	ierr = VecAssemblyEnd(funcVecCos);CHKERRQ(ierr);

	if (sectionDisp) {
		ierr = PetscPrintf(comm,"%s Petsc Section View %s\n", bar, bar);CHKERRQ(ierr);
		ierr = PetscSectionView(section, 0);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s End Petsc Section View %s\n",bar, bar);CHKERRQ(ierr);
	}

	if (VTKdisp) {
		PetscViewer	vtkviewerpart;
		Vec 		partition;
		DM		dmLocal;

		ierr = VTKPartitionVisualize(dm, &dmLocal, &partition);CHKERRQ(ierr);
		ierr = PetscViewerCreate(comm, &vtkviewerpart);CHKERRQ(ierr);
		ierr = PetscViewerSetType(vtkviewerpart,PETSCVIEWERVTK);CHKERRQ(ierr);
		ierr = PetscViewerFileSetName(vtkviewerpart, "partition-map.vtk");CHKERRQ(ierr);
		ierr = VecView(partition, vtkviewerpart);CHKERRQ(ierr);
		ierr = PetscViewerDestroy(&vtkviewerpart);CHKERRQ(ierr);
		ierr = VecDestroy(&partition);CHKERRQ(ierr);
		ierr = DMDestroy(&dmLocal);CHKERRQ(ierr);
	}

	/*	LOOP OVER ALL VERTICES ON LOCAL MESH	*/
	if (arrayDisp) {PetscPrintf(comm,"%s Array %s\n",bar, bar);
                ierr = PetscPrintf(comm, "Before Op | After Op\n");CHKERRQ(ierr);
        }
        for(j = 0; j < nVertex; ++j) {
		if (arrayDisp) {ierr = PetscPrintf(comm, "%.3f", array[j]);CHKERRQ(ierr);}
                ierr = VecDot(funcVecCos, funcVecSin, &dot);CHKERRQ(ierr);
		array[j] = dot;
                if (arrayDisp) {ierr = PetscPrintf(comm, "\t  |%.3f\n", array[j]);CHKERRQ(ierr);}
        }

	if (arrayDisp) {
		ierr = PetscPrintf(comm,"%d Number of LOCAL elements\n", nVertex);CHKERRQ(ierr);
        	ierr = PetscPrintf(comm,"%s Array End %s\n", bar, bar);CHKERRQ(ierr);
	}

	/*	Put LOCAL with changed values back into GLOBAL	*/
	ierr = VecRestoreArray(solVecLocal, &array);CHKERRQ(ierr);
	ierr = VecDestroy(&funcVecSin);CHKERRQ(ierr);
	ierr = VecDestroy(&funcVecCos);CHKERRQ(ierr);
	ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);

        /*	Init Log	*/
	ierr = PetscLogStageRegister("Commun", &stage);CHKERRQ(ierr);
	ierr = PetscLogEventRegister("CommuE", 0, &event);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
	ierr = PetscLogEventBegin(event, 0, 0, 0, 0);CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
        /*	Push LocalToGlobal time to log	*/
        ierr = PetscLogEventEnd(event, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = PetscLogStagePop();CHKERRQ(ierr);

	/*	Get LOCAL coordinates for debug	*/
	ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
	ierr = VecGetLocalSize(coordinates, &nCoords);CHKERRQ(ierr);
	ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);

	/*	LOOP OVER ALL COORDINATES PAIRS ON LOCAL MESH
		NOTE: This is not the same as looping over values of a matrix A
		representing the "vertices" but instead gives you the (x,y)
		coordinates corresponding to an entry Aij. Rule of thumb for checking
		is that there should be twice as many local coords as local vertices!	*/
	if (coordDisp) {
		ierr = PetscPrintf(comm,"%s Coords %s\n", bar, bar);CHKERRQ(ierr);
		for(i=0; i < nCoords/2; i++) {
			ierr = PetscPrintf(comm,"(%.2f,%.2f)\n", coords[2*i], coords[(2*i)+1]);CHKERRQ(ierr);
		}
		ierr = PetscPrintf(comm,"%d Number of LOCAL coordinates\n",i);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s Coords End %s\n", bar, bar);CHKERRQ(ierr);
	}

	/*	Output vtk of global solution vector	*/
	if (VTKdisp) {
		PetscViewer	vtkviewersoln;

		ierr = PetscViewerCreate(comm, &vtkviewersoln);CHKERRQ(ierr);
		ierr = PetscViewerSetType(vtkviewersoln,PETSCVIEWERVTK);CHKERRQ(ierr);
		ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk");CHKERRQ(ierr);
		ierr = VecView(solVecGlobal, vtkviewersoln);CHKERRQ(ierr);
		ierr = PetscViewerDestroy(&vtkviewersoln);CHKERRQ(ierr);
	}

	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	ierr = VecDestroy(&solVecGlobal);CHKERRQ(ierr);
	ierr = DMDestroy(&dm);CHKERRQ(ierr);
	ierr = PetscFinalize();CHKERRQ(ierr);
	return ierr;
}
