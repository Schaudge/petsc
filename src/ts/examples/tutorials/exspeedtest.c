static char help[33] = "Test Unstructured Mesh Handling\n";

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

PetscErrorCode GeneralInfo(MPI_Comm comm, char *bar, PetscViewer genViewer)
{
	PetscErrorCode	ierr;
	const char 	*string;

	ierr = PetscPrintf(comm, "%s General Info %s\n", bar + 2, bar + 2);CHKERRQ(ierr);
	ierr = PetscViewerStringGetStringRead(genViewer, &string, NULL);CHKERRQ(ierr);
	ierr = PetscPrintf(comm, string);CHKERRQ(ierr);
	ierr = PetscPrintf(comm, "%s End General Info %s\n", bar + 2, bar + 5);CHKERRQ(ierr);

	return ierr;
}

/* 	Main	*/
int main(int argc, char **argv)
{
	MPI_Comm		comm;
	PetscErrorCode		ierr;
	PetscViewer		genViewer;
	PetscPartitioner	partitioner;
	PetscPartitionerType	partitionername;
	PetscLogStage 		stageINSERT, stageADD, stageGVD;
	PetscLogEvent 		eventINSERT, eventADD, eventGVD;
	DM			dm, dmDist, dmInterp;
	IS			bcPointsIS, globalCellNumIS;
	PetscSection		section;
	Vec			funcVecSin, funcVecCos, solVecLocal, solVecGlobal, coordinates, VDot;
	PetscBool		perfTest = PETSC_FALSE, fileflg = PETSC_FALSE, dmDistributed = PETSC_FALSE, dmInterped = PETSC_TRUE, dispFlag = PETSC_FALSE, isView = PETSC_FALSE,  VTKdisp = PETSC_FALSE, dmDisp = PETSC_FALSE, sectionDisp = PETSC_FALSE, arrayDisp = PETSC_FALSE, coordDisp = PETSC_FALSE;
	PetscInt		dim = 2, overlap = 0, meshSize = 10, i, j, k, numFields = 100, numBC = 1, vecsize = 1000, nCoords, nVertex, globalSize, globalCellSize, commiter;
	PetscInt		bcField[numBC];
        size_t                  namelen=0;
	PetscScalar 		dot, VDotResult;
	PetscScalar		*coords, *array;
	char			genInfo[PETSC_MAX_PATH_LEN]="", bar[20] = "-----------------\0", filename[PETSC_MAX_PATH_LEN]="";

	ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr) return ierr;
	comm = PETSC_COMM_WORLD;
	ierr = PetscViewerStringOpen(comm, genInfo, sizeof(genInfo), &genViewer);CHKERRQ(ierr);

	ierr = PetscOptionsBegin(comm, NULL, "Speedtest Options", "");CHKERRQ(ierr); {
		ierr = PetscOptionsBool("-speed", "Streamline program to only perform necessary operations for performance testing", "", perfTest, &perfTest, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-vtkout", "Enable mesh distribution visualization", "", VTKdisp, &VTKdisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-disp", "Turn on all displays", "", dispFlag, &dispFlag, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-isview", "Turn on ISView for single threaded", "", isView, &isView, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-dmview", "Turn on DMView", "", dmDisp, &dmDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-secview","Turn on SectionView", "", sectionDisp, &sectionDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-arrview", "Turn on array display", "", arrayDisp, &arrayDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsBool("-coordview","Turn on coordinate display", "", coordDisp, &coordDisp, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &fileflg); CHKERRQ(ierr);
		ierr = PetscOptionsGetInt(NULL, NULL, "-n", &meshSize, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsGetInt(NULL, NULL, "-nf", &numFields, NULL);CHKERRQ(ierr);
		ierr = PetscOptionsGetInt(NULL, NULL, "-overlap", &overlap, NULL);CHKERRQ(ierr);
	}
	ierr = PetscOptionsEnd();CHKERRQ(ierr);
	if (dispFlag) {isView = PETSC_TRUE; dmDisp = PETSC_TRUE; sectionDisp = PETSC_TRUE, arrayDisp = PETSC_TRUE; coordDisp = PETSC_TRUE;}

	PetscInt		numDOF[numFields*(dim+1)], numComp[numFields], faces[dim];
        ierr = PetscStrlen(filename, &namelen);CHKERRQ(ierr);
        if (!namelen){

		for(i = 0; i < dim; i++){
			/* Make the default box mesh creation with CLI options	*/
			faces[i] = meshSize;
		}
          	ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, faces, NULL, NULL, NULL, dmInterped, &dm);CHKERRQ(ierr);
        } else {
          	ierr = DMPlexCreateFromFile(comm, filename, dmInterped, &dm);CHKERRQ(ierr);
		ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
	}

	ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		dm = dmDist;
		dmDistributed = PETSC_TRUE;
	}else{
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
			dmInterped = PETSC_TRUE;
		}else{
			ierr = PetscPrintf(comm,"No interped dm [QUITE UNUSUAL]\n");CHKERRQ(ierr);
		}
	}

	/* 	Init number of Field Components	*/
	for (k = 0; k < numFields; k++){numComp[k] = 1;}
	/*	Init numDOF[field componentID] = Not Used	*/
	for (k = 0; k < numFields*(dim+1); ++k){numDOF[k] = 0;}
	/*	numDOF[field componentID] = Used	*/
	numDOF[0] = 1;
	/*	bcField[boundary conditionID] = Dirichtlet Val	*/
	bcField[0] = 0;

	/*	Assign BC using IS of LOCAL boundaries	*/
        ierr = DMGetStratumIS(dm, "depth", 2, &bcPointsIS);CHKERRQ(ierr);
	ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
	ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
	ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
	ierr = DMSetSection(dm, section);CHKERRQ(ierr);
	if (sectionDisp) {
		ierr = PetscPrintf(comm,"%s Petsc Section View %s\n", bar, bar);CHKERRQ(ierr);
		ierr = PetscSectionView(section, 0);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s End Petsc Section View %s\n",bar, bar);CHKERRQ(ierr);
	}
	ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
	ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);
	if (dmDisp) {
		ierr = PetscPrintf(comm,"%s DM View %s\n", bar, bar);CHKERRQ(ierr);
		ierr = DMView(dm, 0);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s End DM View %s\n", bar, bar);CHKERRQ(ierr);
	}

	/*	Create Vector for per process function evaluation	*/
	if (!perfTest){
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

	/*	LOOP OVER ALL VERTICES ON LOCAL MESH UNLESS ITS A SPEEDTEST */
	if (!perfTest) {
		/*	Perform Function on LOCAL array	*/
		ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
		ierr = VecGetLocalSize(solVecLocal, &nVertex);CHKERRQ(ierr);
		ierr = VecGetArray(solVecLocal, &array);CHKERRQ(ierr);
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
        	ierr = VecDestroy(&funcVecSin);CHKERRQ(ierr);
        	ierr = VecDestroy(&funcVecCos);CHKERRQ(ierr);
		ierr = VecRestoreArray(solVecLocal, &array);CHKERRQ(ierr);
		ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
        }

	/*	Perform setup before timing	*/
	ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
	ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
	ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);

	/*	Init INSERT_VALUES timing only log	*/
	ierr = PetscLogStageRegister("CommStageINSERT", &stageINSERT);CHKERRQ(ierr);
	ierr = PetscLogEventRegister("CommINSERT", 0, &eventINSERT);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stageINSERT);CHKERRQ(ierr);
	ierr = PetscLogEventBegin(eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
        for (commiter = 0; commiter < 100; commiter++) {
        	ierr = DMLocalToGlobalBegin(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
                ierr = DMLocalToGlobalEnd(dm, solVecLocal, INSERT_VALUES, solVecGlobal);CHKERRQ(ierr);
		ierr = DMGlobalToLocalBegin(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
		ierr = DMGlobalToLocalEnd(dm, solVecGlobal, INSERT_VALUES, solVecLocal);CHKERRQ(ierr);
        }
	/*	Push LocalToGlobal time to log	*/
	ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(eventINSERT, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = PetscLogStagePop();CHKERRQ(ierr);

	/*	Perform setup before timing	*/
	ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = DMGetLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
	ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
	ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);

        /*	Init ADD_VALUES Log	*/
	ierr = PetscLogStageRegister("CommStageADDVAL", &stageADD);CHKERRQ(ierr);
	ierr = PetscLogEventRegister("CommADDVAL", 0, &eventADD);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stageADD);CHKERRQ(ierr);
	ierr = PetscLogEventBegin(eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
        for (commiter = 0; commiter < 100; commiter++) {
          	ierr = DMLocalToGlobalBegin(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
                ierr = DMLocalToGlobalEnd(dm, solVecLocal, ADD_VALUES, solVecGlobal);CHKERRQ(ierr);
		/*	These aren't implemented	*/
		//		ierr = DMGlobalToLocalBegin(dm, solVecGlobal, ADD_VALUES, solVecLocal);CHKERRQ(ierr);
		//		ierr = DMGlobalToLocalEnd(dm, solVecGlobal, ADD_VALUES, solVecLocal);CHKERRQ(ierr);
        }
        /*	Push time to log	*/
	ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = DMRestoreLocalVector(dm, &solVecLocal);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(eventADD, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = PetscLogStagePop();CHKERRQ(ierr);

	/*	Perform setup before timing	*/
	ierr = DMCreateGlobalVector(dm, &VDot);CHKERRQ(ierr);
	ierr = VecSet(VDot, 1);CHKERRQ(ierr);
	ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
	ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);

	/*	Init VecDot Log	*/
	ierr = PetscLogStageRegister("CommStageGlblVecDot", &stageGVD);CHKERRQ(ierr);
	ierr = PetscLogEventRegister("CommGlblVecDot", 0, &eventGVD);CHKERRQ(ierr);
	ierr = PetscLogStagePush(stageGVD);CHKERRQ(ierr);
	ierr = PetscLogEventBegin(eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
	for (commiter = 0; commiter < 100; commiter++) {
		ierr = VecDotBegin(VDot, VDot, &VDotResult);CHKERRQ(ierr);
		ierr = VecDotEnd(VDot, VDot, &VDotResult);CHKERRQ(ierr);
        }
	/*	Push time to log	*/
	ierr = VecDestroy(&VDot);CHKERRQ(ierr);
	ierr = PetscLogEventEnd(eventGVD, 0, 0, 0, 0);CHKERRQ(ierr);
        ierr = PetscLogStagePop();CHKERRQ(ierr);

	if (coordDisp) {
		/*	Get LOCAL coordinates for debug	*/
		ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
		ierr = VecGetLocalSize(coordinates, &nCoords);CHKERRQ(ierr);
		ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);

		/*	LOOP OVER ALL COORDINATES PAIRS ON LOCAL MESH
		NOTE: This is not the same as looping over values of a matrix A
		representing the "vertices" but instead gives you the (x,y)
		coordinates corresponding to an entry Aij. Rule of thumb for checking
		is that there should be twice as many local coords as local vertices!	*/

		ierr = PetscPrintf(comm,"%s Coords %s\n", bar, bar);CHKERRQ(ierr);
		for(i=0; i < nCoords/2; i++) {
			ierr = PetscPrintf(comm,"(%.2f,%.2f)\n", coords[2*i], coords[(2*i)+1]);CHKERRQ(ierr);
		}
		ierr = PetscPrintf(comm,"%d Number of LOCAL coordinates\n",i);CHKERRQ(ierr);
		ierr = PetscPrintf(comm,"%s Coords End %s\n", bar, bar);CHKERRQ(ierr);
		ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
	}

	/*	Output vtk of global solution vector	*/
	if (VTKdisp) {
		PetscViewer	vtkviewersoln;

		ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
		ierr = PetscViewerCreate(comm, &vtkviewersoln);CHKERRQ(ierr);
		ierr = PetscViewerSetType(vtkviewersoln,PETSCVIEWERVTK);CHKERRQ(ierr);
		ierr = PetscViewerFileSetName(vtkviewersoln, "solution.vtk");CHKERRQ(ierr);
		ierr = VecView(solVecGlobal, vtkviewersoln);CHKERRQ(ierr);
		ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
		ierr = PetscViewerDestroy(&vtkviewersoln);CHKERRQ(ierr);
	}

	/*	Get Some additional data about the mesh mainly for printing */
	ierr = DMGetGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = VecGetSize(solVecGlobal, &globalSize);CHKERRQ(ierr);
	ierr = DMRestoreGlobalVector(dm, &solVecGlobal);CHKERRQ(ierr);
	ierr = DMPlexGetCellNumbering(dm, &globalCellNumIS);CHKERRQ(ierr);
	ierr = ISGetSize(globalCellNumIS, &globalCellSize);CHKERRQ(ierr);
	ierr = DMPlexGetPartitioner(dm, &partitioner);CHKERRQ(ierr);CHKERRQ(ierr);
	ierr = PetscPartitionerGetType(partitioner, &partitionername);CHKERRQ(ierr);

	/*	Aggregate all of the information for printing	*/
	{
	ierr = PetscViewerStringSPrintf(genViewer, "Partitioner Used:%s>%s\n", bar + 2, partitionername);CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(genViewer, "Global Node Num:%s>%d\n", bar + 1, globalSize);CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(genViewer, "Global Cell Num:%s>%d\n", bar + 1, globalCellSize);CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(genViewer, "Dimension of mesh:%s>%d\n", bar + 3, dim);CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(genViewer, "Number of Fields:%s>%d", bar + 2, numFields);CHKERRQ(ierr);
	if (numFields == 100) {
		ierr = PetscViewerStringSPrintf(genViewer, "(default)\n");CHKERRQ(ierr);
	} else {
		ierr = PetscViewerStringSPrintf(genViewer, "\n");CHKERRQ(ierr);
	}
	ierr = PetscViewerStringSPrintf(genViewer, "Ghost point overlap:%s>%d\n", bar + 5, overlap);CHKERRQ(ierr);

	ierr = PetscViewerStringSPrintf(genViewer, "\nFile read mode:%s>%s\n", bar, fileflg ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	if (fileflg) {
		ierr = PetscViewerStringSPrintf(genViewer, "┗ File read name:%s>%s\n", bar + 2, filename);CHKERRQ(ierr);
	}
	ierr = PetscViewerStringSPrintf(genViewer,  "Distributed dm:%s>%s\n", bar, dmDistributed ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerStringSPrintf(genViewer, "Interpolated dm:%s>%s\n", bar + 1, dmInterped ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "Performance test mode:%s>%s\n", bar + 7, perfTest ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "VTKoutput mode:%s>%s\n", bar, VTKdisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "Full Display mode:%s>%s\n", bar + 3, dispFlag ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "IS Display mode:%s>%s\n", bar + 1, isView ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "DM Display mode:%s>%s\n", bar + 1, dmDisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "Section Display mode:%s>%s\n", bar + 6, sectionDisp? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "Array Display mode:%s>%s\n", bar + 4, arrayDisp ? "PETSC_TRUE * " : "PETSC_FALSE");CHKERRQ(ierr);
        ierr = PetscViewerStringSPrintf(genViewer, "Coord Disp mode:%s>%s\n", bar + 1, coordDisp ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	}

	ierr = GeneralInfo(comm, bar, genViewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
	ierr = DMDestroy(&dm);CHKERRQ(ierr);
	ierr = PetscFinalize();CHKERRQ(ierr);
	return ierr;
}
