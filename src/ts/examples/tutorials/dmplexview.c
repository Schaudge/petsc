static char help[33] = "Concise View of DMPlex Object\n";

# include <petscdmplex.h>
# include <petsc/private/dmpleximpl.h>

PetscErrorCode DMPlexGetEdgeNumbering(DM dm, IS *globalEdgeNumbers)
{
	PetscErrorCode	ierr;
	PetscSection   	section, globalSection;
	PetscInt      	pStart, pEnd, shift=0, *numbers, p;

	ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
	ierr = DMPlexGetDepthStratum(dm, 1, &pStart, &pEnd);CHKERRQ(ierr);
	ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
	for (p = pStart; p < pEnd; ++p) {
		ierr = PetscSectionSetDof(section, p, 1);CHKERRQ(ierr);
	}
	ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
	ierr = PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
	ierr = PetscMalloc1(pEnd - pStart, &numbers);CHKERRQ(ierr);
	for (p = pStart; p < pEnd; ++p) {
		ierr = PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]);CHKERRQ(ierr);
		if (numbers[p-pStart] < 0) numbers[p-pStart] -= shift;
		else                       numbers[p-pStart] += shift;
	}
	ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd - pStart, numbers, PETSC_OWN_POINTER, globalEdgeNumbers);CHKERRQ(ierr);
	if (NULL) {
		PetscLayout layout;
		ierr = PetscSectionGetPointLayout(PetscObjectComm((PetscObject) dm), globalSection, &layout);CHKERRQ(ierr);
		ierr = PetscLayoutGetSize(layout, NULL);CHKERRQ(ierr);
		ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
	}
	ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMPlexGetFaceNumbering(DM dm, IS *globalFaceNumbers)
{
	PetscErrorCode	ierr;
	PetscSection   	section, globalSection;
	PetscInt      	pStart, pEnd, shift=0, *numbers, p;

	ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
	ierr = DMPlexGetDepthStratum(dm, 2, &pStart, &pEnd);CHKERRQ(ierr);
	ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
	for (p = pStart; p < pEnd; ++p) {
		ierr = PetscSectionSetDof(section, p, 1);CHKERRQ(ierr);
	}
	ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
	ierr = PetscSectionCreateGlobalSection(section, dm->sf, PETSC_FALSE, PETSC_FALSE, &globalSection);CHKERRQ(ierr);
	ierr = PetscMalloc1(pEnd - pStart, &numbers);CHKERRQ(ierr);
	for (p = pStart; p < pEnd; ++p) {
		ierr = PetscSectionGetOffset(globalSection, p, &numbers[p-pStart]);CHKERRQ(ierr);
		if (numbers[p-pStart] < 0) numbers[p-pStart] -= shift;
		else                       numbers[p-pStart] += shift;
	}
	ierr = ISCreateGeneral(PetscObjectComm((PetscObject) dm), pEnd - pStart, numbers, PETSC_OWN_POINTER, globalFaceNumbers);CHKERRQ(ierr);
	if (NULL) {
		PetscLayout layout;
		ierr = PetscSectionGetPointLayout(PetscObjectComm((PetscObject) dm), globalSection, &layout);CHKERRQ(ierr);
		ierr = PetscLayoutGetSize(layout, NULL);CHKERRQ(ierr);
		ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
	}
	ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&globalSection);CHKERRQ(ierr);

	return ierr;
}

PetscErrorCode DMPlexGetXXXPerProcess(DM dm, PetscInt depth, PetscInt *numBins, PetscScalar *numPerProcess[], PetscInt  *binnedProcesses[])
{
	MPI_Comm		comm;
	PetscMPIInt		rank, size;
	PetscErrorCode		ierr;
	IS			PerProcessTaggedIS;
	ISLocalToGlobalMapping	ltog;
	Vec 			vecPerProcess;
	VecTagger		tagger;
	VecTaggerBox   		*box;
	PetscInt		i, cstart, cend, numBins_;
	PetscInt		*binnedProcesses_;
	PetscScalar		*numPerProcess_;

	ierr = PetscObjectGetComm((PetscObject) dm, &comm);
	ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
	ierr = VecCreateMPI(comm, 1, size, &vecPerProcess);CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingCreate(comm, 1, 1, &rank, PETSC_COPY_VALUES, &ltog);CHKERRQ(ierr);
	ierr = VecSetLocalToGlobalMapping(vecPerProcess, ltog);CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
	ierr = DMPlexGetDepthStratum(dm, depth, &cstart, &cend);CHKERRQ(ierr);
	ierr = VecSetValueLocal(vecPerProcess, 0, cend-cstart, INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecUniqueEntries(vecPerProcess, &numBins_, &numPerProcess_);CHKERRQ(ierr);
	ierr = PetscCalloc1(numBins_, &binnedProcesses_);CHKERRQ(ierr);
	for (i = 0; i < numBins_; i++) {
		ierr = VecTaggerCreate(comm, &tagger);CHKERRQ(ierr);
		ierr = VecTaggerSetType(tagger, VECTAGGERABSOLUTE);CHKERRQ(ierr);
		ierr = PetscMalloc1(1, &box);CHKERRQ(ierr);
		box->min = numPerProcess_[i]-0.5;
		box->max = numPerProcess_[i]+0.5;
		ierr = VecTaggerAbsoluteSetBox(tagger, box);CHKERRQ(ierr);
		ierr = PetscFree(box);CHKERRQ(ierr);
		ierr = VecTaggerSetUp(tagger);CHKERRQ(ierr);
		ierr = VecTaggerComputeIS(tagger, vecPerProcess, &PerProcessTaggedIS);CHKERRQ(ierr);
		ierr = ISGetSize(PerProcessTaggedIS, &binnedProcesses_[i]);CHKERRQ(ierr);
		ierr = ISDestroy(&PerProcessTaggedIS);CHKERRQ(ierr);
		ierr = VecTaggerDestroy(&tagger);CHKERRQ(ierr);
	}
	ierr = VecDestroy(&vecPerProcess);CHKERRQ(ierr);
	if (numBins) *numBins = numBins_;
	if (numPerProcess) { *numPerProcess = numPerProcess_;} else { ierr = PetscFree(numPerProcess_);CHKERRQ(ierr);}
	*binnedProcesses = binnedProcesses_;
	return ierr;
}

PetscErrorCode DMPlexViewAsciiConcise(DM dm, PetscViewer viewer)
{
	MPI_Comm		comm;
	PetscMPIInt		rank = 0, size = 0;
	PetscErrorCode		ierr;
	IS			vertexIS = NULL, edgeIS = NULL, faceIS = NULL, cellIS = NULL;
	PetscBool		dmDistributed = PETSC_FALSE, dmInterped = PETSC_FALSE, facesOK = PETSC_FALSE, symmetryOK = PETSC_FALSE, skeletonOK = PETSC_FALSE, pointSFOK = PETSC_FALSE, geometryOK = PETSC_FALSE, coneConformOnInterfacesOK = PETSC_FALSE;
	PetscInt		i, locdepth, depth, globalVertexSize = 0, globalEdgeSize = 0, globalFaceSize = 0, globalCellSize = 0, dim = 0, numBinnedVertexProcesses, numBinnedEdgeProcesses, numBinnedFaceProcesses, numBinnedCellProcesses;
	PetscInt		*binnedVertices = NULL, *binnedEdges = NULL, *binnedFaces = NULL, *binnedCells = NULL;
	PetscScalar		 *verticesPerProcess = NULL, *edgesPerProcess = NULL, *facesPerProcess = NULL, *cellsPerProcess = NULL;
	char                    bar[19] = "-----------------\0";

	/* Init	*/
	if (!viewer) {
		PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) dm), &viewer);
	}
	ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
	ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dm), &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dm), &size);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "%s General Info %s\n", bar + 2, bar + 2);CHKERRQ(ierr);
	if (size > 1) {
		PetscSF		sf;

		ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
		if (sf) {
			//ierr = DMPlexDistributeOverlap(dm, 1, NULL, &dm);CHKERRQ(ierr);
			dmDistributed = PETSC_TRUE;
		}
	}

	/* Global and Local Sizes	*/
	ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
	ierr = DMPlexGetDepth(dm, &locdepth);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locdepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	if (dim == depth) {
		dmInterped = PETSC_TRUE;
	}
	for (i = 0; i <= depth; i++) {
		if (!dmInterped && (i != 0)) { i = 3;}
		/* In case that dm is not interpolated, will only have cell-vertex mesh	*/
		if ((dim == 2) && (i == depth)) { i = 3;}
		/* For 2D calls the faces "cells"	*/
		PetscInt 	min = 0, max = 0;
		switch (i)
			{
			case 0:
				ierr = DMPlexGetVertexNumbering(dm, &vertexIS);CHKERRQ(ierr);
				ierr = ISGetMinMax(vertexIS, &min, &max);CHKERRQ(ierr);
				max = PetscAbsInt(max); min = PetscAbsInt(min);
				max = PetscMax(max, min);
				if (size < 2) { max += 1;} //0 indexing strikes again
				ierr = MPI_Reduce(&max, &globalVertexSize, 1, MPIU_INT, MPI_MAX, 0, comm);CHKERRQ(ierr);
				ierr = DMPlexGetXXXPerProcess(dm, i, &numBinnedVertexProcesses, &verticesPerProcess, &binnedVertices);CHKERRQ(ierr);
				break;
			case 1:
				ierr = DMPlexGetEdgeNumbering(dm, &edgeIS);CHKERRQ(ierr);
				ierr = ISGetMinMax(edgeIS, &min, &max);CHKERRQ(ierr);
				max = PetscAbsInt(max); min = PetscAbsInt(min);
				max = PetscMax(max, min);
				if (size < 2) { max += 1;} //0 indexing strikes again
				ierr = MPI_Reduce(&max, &globalEdgeSize, 1, MPIU_INT, MPI_MAX, 0, comm);CHKERRQ(ierr);
				ierr = ISDestroy(&edgeIS);CHKERRQ(ierr);
				ierr = DMPlexGetXXXPerProcess(dm, i, &numBinnedEdgeProcesses, &edgesPerProcess, &binnedEdges);CHKERRQ(ierr);
				break;
			case 2:
				ierr = DMPlexGetFaceNumbering(dm, &faceIS);CHKERRQ(ierr);
				ierr = ISGetMinMax(faceIS, &min, &max);CHKERRQ(ierr);
				max = PetscAbsInt(max); min = PetscAbsInt(min);
				max = PetscMax(max, min);
				if (size < 2) { max += 1;} //0 indexing strikes again
				ierr = MPI_Reduce(&max, &globalFaceSize, 1, MPIU_INT, MPI_MAX, 0, comm);CHKERRQ(ierr);
				ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
				ierr = DMPlexGetXXXPerProcess(dm, i, &numBinnedFaceProcesses, &facesPerProcess, &binnedFaces);CHKERRQ(ierr);
                                break;
			case 3:
				ierr = DMPlexGetCellNumbering(dm, &cellIS);CHKERRQ(ierr);
				ierr = ISGetMinMax(cellIS, &min, &max);CHKERRQ(ierr);
				max = PetscAbsInt(max); min = PetscAbsInt(min);
				max = PetscMax(max, min);
				if (size < 2) { max += 1;} //0 indexing strikes again
				ierr = MPI_Reduce(&max, &globalCellSize, 1, MPIU_INT, MPI_MAX, 0,  comm);CHKERRQ(ierr);
				if (dim == 2) i = 2; //result of hacking faces = cells
				ierr = DMPlexGetXXXPerProcess(dm, i, &numBinnedCellProcesses, &cellsPerProcess, &binnedCells);CHKERRQ(ierr);
                                break;
			default:
				ierr = PetscPrintf(comm, "%i depth not suppoerted\n", i);CHKERRQ(ierr);
				break;
			}
	}

	{
	/* Multiplicity
	ierr = DMGetStratumIS(dm, "depth", 0, &vertexIS);CHKERRQ(ierr);
	ierr = ISGetIndices(vertexIS, &vertexidx);CHKERRQ(ierr);
	ierr = ISGetSize(vertexIS, &localVertexSize);CHKERRQ(ierr);
	ierr = DMPlexGetMaxSizes(dm, NULL, &locmaxSupportSize);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmaxSupportSize, &maxSupportSize, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, maxSupportSize+1, &multiplicity);CHKERRQ(ierr);
	ierr = VecSet(multiplicity, 0);CHKERRQ(ierr);
	for (i = 0; i < localVertexSize; i++) {
		const PetscInt	point = vertexidx[i];
		ierr = DMPlexGetSupportSize(dm, point, &localSupportSize);CHKERRQ(ierr);
		ierr = VecSetValue(multiplicity, localSupportSize, 1, ADD_VALUES);CHKERRQ(ierr);
		if (localSupportSize > locmaxMultiplicity) {
			locmaxMultiplicity = localSupportSize;
		}
		if (localSupportSize < locminMultiplicity) {
			locminMultiplicity = localSupportSize;
		}
	}
	ierr = ISRestoreIndices(vertexIS, &vertexidx);CHKERRQ(ierr);
	ierr = ISDestroy(&vertexIS);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmaxMultiplicity, &maxMultiplicity, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locminMultiplicity, &minMultiplicity, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	PetscInt	globalmultiplicities[maxSupportSize+1];
	for (i = 0; i < maxSupportSize+1; i++) {
		globalmultiplicities[i] = 0.0;
	}
	ierr = VecGetArray(multiplicity, &locmultiplicities);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmultiplicities, &globalmultiplicities, maxSupportSize, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);
	*/}

	/* Various Diagnostic DMPlex Checks	*/
	ierr = DMPlexCheckFaces(dm, 0);CHKERRQ(ierr); if (!ierr) { facesOK = PETSC_TRUE;}
	ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr); if (!ierr) { symmetryOK = PETSC_TRUE;}
	ierr = DMPlexCheckSkeleton(dm, 0);CHKERRQ(ierr); if (!ierr) { skeletonOK = PETSC_TRUE;}
	ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr); if(!ierr) { pointSFOK = PETSC_TRUE;}
	ierr = DMPlexCheckGeometry(dm);CHKERRQ(ierr); if (!ierr) { geometryOK = PETSC_TRUE;}
	ierr = DMPlexCheckConesConformOnInterfaces(dm);CHKERRQ(ierr); if(!ierr) { coneConformOnInterfacesOK = PETSC_TRUE;}

	/* Printing	*/
	/* Autotest Output	*/
	{
	ierr = PetscViewerASCIIPrintf(viewer, "Face Orientation OK:%s>%s\n", bar + 5, facesOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Adjacency Symmetry OK:%s>%s\n", bar + 7, symmetryOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Cells Vertex Count OK:%s>%s\n", bar + 7, skeletonOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Point SF OK:%s%s>%s\n", bar, bar + 14, pointSFOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Geometry OK:%s%s>%s\n", bar, bar + 14, geometryOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Cone Interfaces Conform OK:%s>%s\n\n", bar + 12, coneConformOnInterfacesOK ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	}

	/* Mesh Information	*/
	{
	ierr = PetscViewerASCIIPrintf(viewer, "Distributed DM:%s>%s\n", bar, dmDistributed ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Interpolated DM:%s>%s\n", bar + 1, dmInterped ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Dimension of mesh:%s>%d\n", bar + 3, dim);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Global Vertex Num:%s>%d\n", bar + 3, globalVertexSize);CHKERRQ(ierr);
	if (globalEdgeSize > 0) { ierr = PetscViewerASCIIPrintf(viewer, "Global Edge Num:%s>%d\n", bar + 1, globalEdgeSize);CHKERRQ(ierr);}
	if (globalFaceSize > 0) { ierr = PetscViewerASCIIPrintf(viewer, "Global Face Num:%s>%d\n", bar + 1, globalFaceSize);CHKERRQ(ierr);}
	ierr = PetscViewerASCIIPrintf(viewer, "Global Cell Num:%s>%d\n", bar + 1, globalCellSize);CHKERRQ(ierr);
	}

	/*ierr = PetscViewerASCIIPrintf(viewer, "Vertex Multiplicity Range: %d - %d\n", minMultiplicity, maxMultiplicity);CHKERRQ(ierr);
	for (i = minMultiplicity; i <= maxMultiplicity; i++) {
		if (i == minMultiplicity) {
			//ierr = PetscViewerASCIIPrintf(viewer, "Multipl:NumVert\n");CHKERRQ(ierr);
		}
		//		ierr = PetscViewerASCIIPrintf(viewer, "\t%d - %d\n", i, globalmultiplicities[i]);CHKERRQ(ierr);
		}*/

	/* Parallel Information	*/
	{
	if (binnedVertices) {
		ierr = PetscViewerASCIIPrintf(viewer, "\nVertices Per Process Range: %.0f - %.0f\n", verticesPerProcess[0], verticesPerProcess[numBinnedVertexProcesses-1]);CHKERRQ(ierr);
		for (i = 0; i < numBinnedVertexProcesses; i++) {
			if (i == 0) {
				ierr = PetscViewerASCIIPrintf(viewer, "Num Per Proc. - Num Proc.\n");CHKERRQ(ierr);
			}
			ierr = PetscViewerASCIIPrintf(viewer, "\t%5.0f - %d\n", verticesPerProcess[i], binnedVertices[i]);CHKERRQ(ierr);
		}
	}
	if (binnedEdges) {
		ierr = PetscViewerASCIIPrintf(viewer, "\nEdges Per Process Range: %.0f - %.0f\n", edgesPerProcess[0], edgesPerProcess[numBinnedEdgeProcesses-1]);CHKERRQ(ierr);
		for (i = 0; i < numBinnedEdgeProcesses; i++) {
			if (i == 0) {
				ierr = PetscViewerASCIIPrintf(viewer, "Num Per Proc. - Num Proc.\n");CHKERRQ(ierr);
			}
			ierr = PetscViewerASCIIPrintf(viewer, "\t%5.0f - %d\n", edgesPerProcess[i], binnedEdges[i]);CHKERRQ(ierr);
		}
	}
	if (binnedFaces) {
		ierr = PetscViewerASCIIPrintf(viewer, "\nFaces Per Process Range: %.0f - %.0f\n", facesPerProcess[0], facesPerProcess[numBinnedFaceProcesses-1]);CHKERRQ(ierr);
		for (i = 0; i < numBinnedFaceProcesses; i++) {
			if (i == 0) {
				ierr = PetscViewerASCIIPrintf(viewer, "Num Per Proc. - Num Proc.\n");CHKERRQ(ierr);
			}
			ierr = PetscViewerASCIIPrintf(viewer, "\t%5.0f - %d\n", facesPerProcess[i], binnedFaces[i]);CHKERRQ(ierr);
		}
	}
	if (binnedCells) {
		ierr = PetscViewerASCIIPrintf(viewer, "\nCells Per Process Range: %.0f - %.0f\n", cellsPerProcess[0], cellsPerProcess[numBinnedCellProcesses-1]);CHKERRQ(ierr);
		for (i = 0; i < numBinnedCellProcesses; i++) {
			if (i == 0) {
				ierr = PetscViewerASCIIPrintf(viewer, "Num Per Proc. - Num Proc.\n");CHKERRQ(ierr);
			}
			ierr = PetscViewerASCIIPrintf(viewer, "\t%5.0f - %d\n", cellsPerProcess[i], binnedCells[i]);CHKERRQ(ierr);
		}
	}
	}

	ierr = PetscViewerASCIIPrintf(viewer, "%s End General Info %s\n", bar + 2, bar + 5);CHKERRQ(ierr);
	//ierr = VecRestoreArray(multiplicity, &locmultiplicities);CHKERRQ(ierr);
	//ierr = VecDestroy(&multiplicity);CHKERRQ(ierr);
	//if (dmDistributed) { ierr = DMDestroy(&dm);CHKERRQ(ierr);}
	return ierr;
}

int main(int argc, char **argv)
{
	MPI_Comm		comm;
	PetscErrorCode          ierr;
	DM			dm, dmDist;
	IS			bcPointsIS;
	PetscSection		section;
	PetscViewer		genViewer;
	PetscInt		overlap = 0, i, dim = 3, numFields = 1, numBC = 1, faces[dim], bcField[numBC];
	PetscBool		simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

	ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr){ return ierr;}
        comm = PETSC_COMM_WORLD;

	for (i = 0; i < dim; i++) {
		faces[i] = 5;
	}
	ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, dmInterped, &dm);CHKERRQ(ierr);
	//ierr = DMPlexCreateFromFile(comm, "./meshes/2Drectq4_4x4.exo", dmInterped, &dm);CHKERRQ(ierr);
	ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		dm = dmDist;
	}
	ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

	PetscInt	numDOF[numFields*(dim+1)], numComp[numFields];
	for (i = 0; i < numFields; i++){numComp[i] = 1;}
	for (i = 0; i < numFields*(dim+1); i++){numDOF[i] = 0;}
	numDOF[0] = 1;
	bcField[0] = 0;
	ierr = DMGetStratumIS(dm, "depth", dim, &bcPointsIS);CHKERRQ(ierr);
        ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
	ierr = DMPlexCreateSection(dm, NULL, numComp, numDOF, numBC, bcField, NULL, &bcPointsIS, NULL, &section);CHKERRQ(ierr);
        ierr = PetscSectionSetFieldName(section, 0, "Default_Field");CHKERRQ(ierr);
        ierr = DMSetSection(dm, section);CHKERRQ(ierr);
	ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
        ierr = ISDestroy(&bcPointsIS);CHKERRQ(ierr);

	ierr = PetscViewerCreate(comm, &genViewer);CHKERRQ(ierr);
	ierr = PetscViewerSetType(genViewer, PETSCVIEWERASCII);CHKERRQ(ierr);
	ierr = DMPlexViewAsciiConcise(dm, genViewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&genViewer);CHKERRQ(ierr);
	ierr = DMDestroy(&dm);CHKERRQ(ierr);
	ierr = PetscFinalize();CHKERRQ(ierr);
	return ierr;
}
//todo: cell size, binning otrthognal quality, length of boundary layers, how many
//boundary edges
