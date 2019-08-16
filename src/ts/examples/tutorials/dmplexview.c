static char help[33] = "Consice View of DMPlex Object\n";

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

PetscErrorCode DMPlexViewAsciiConcise(DM dmin, PetscViewer viewer)
{
	MPI_Comm		comm;
	PetscMPIInt		rank = 0, size = 0;
	PetscErrorCode		ierr;
	DM			dm;
	IS			vertexIS, edgeIS, faceIS, cellIS, cellPerNodeTaggedIS;
	ISLocalToGlobalMapping	ltog;
	Vec			multiplicity, nodesPerProcess;
	VecTagger		tagger;
	VecTaggerBox   		*box;
	PetscBool		dmDistributed = PETSC_FALSE, dmInterped = PETSC_FALSE;
	PetscInt		i, cstart, cend, locdepth, depth, numBinnedNodes = 0, localVertexSize = 0, globalVertexSize = 0, globalEdgeSize = 0, globalFaceSize = 0, globalCellSize = 0, dim = 0, localSupportSize = 0, locmaxSupportSize, maxSupportSize, locminMultiplicity = 10000, locmaxMultiplicity = 0, minMultiplicity = 10000, maxMultiplicity = 0;
	PetscScalar		*locmultiplicities, *binnedNodes;
	const PetscInt		*vertexidx;
	const char		*name;
	char                    bar[19] = "-----------------\0";

	/* Init	*/
	if (!viewer) {
		PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject) dmin), &viewer);
	}
	ierr = PetscObjectGetComm((PetscObject) dmin, &comm);CHKERRQ(ierr);
	ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) dmin), &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PetscObjectComm((PetscObject) dmin), &size);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "%s General Info %s\n", bar + 2, bar + 2);CHKERRQ(ierr);
	ierr = PetscObjectGetName((PetscObject) dmin, &name);CHKERRQ(ierr);
	if (size > 1) {
		PetscSF		sf, sfDist;

		ierr = DMGetPointSF(dmin, &sf);CHKERRQ(ierr);
		if (sf) {
			dmDistributed = PETSC_TRUE;
			ierr = DMPlexDistributeOverlap(dmin, 1, &sfDist,  &dm);CHKERRQ(ierr);
		} else {
			dm = dmin;
		}
	} else {
		dm = dmin;
	}

	/* Global and Local Sizes	*/
	ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
	ierr = DMPlexGetDepth(dm, &locdepth);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locdepth, &depth, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	if (dim == depth) {
		dmInterped = PETSC_TRUE;
	}
	for (i = 0; i < depth+1; i++) {
		switch (i)
			{
			case 0:
				ierr = DMPlexGetVertexNumbering(dm, &vertexIS);CHKERRQ(ierr);
				ierr = ISGetLocalSize(vertexIS, &globalVertexSize);CHKERRQ(ierr);
				break;
			case 1:
				ierr = DMPlexGetEdgeNumbering(dm, &edgeIS);CHKERRQ(ierr);
				ierr = ISGetLocalSize(edgeIS, &globalEdgeSize);CHKERRQ(ierr);
				break;
			case 2:
				ierr = DMPlexGetFaceNumbering(dm, &faceIS);CHKERRQ(ierr);
				ierr = ISGetLocalSize(faceIS, &globalFaceSize);CHKERRQ(ierr);
                                break;
			case 3:
				ierr = DMPlexGetCellNumbering(dm, &cellIS);CHKERRQ(ierr);
				ierr = ISGetLocalSize(cellIS, &globalCellSize);CHKERRQ(ierr);
                                break;
			default:
				ierr = PetscPrintf(comm, "%i depth not suppoerted\n", i);CHKERRQ(ierr);
				break;
			}
	}

	/* Multiplicity	*/
	ierr = DMGetStratumIS(dm, "depth", 0, &vertexIS);CHKERRQ(ierr);
	ierr = ISGetIndices(vertexIS, &vertexidx);CHKERRQ(ierr);
	ierr = ISGetSize(vertexIS, &localVertexSize);CHKERRQ(ierr);
	ierr = DMPlexGetMaxSizes(dm, NULL, &locmaxSupportSize);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmaxSupportSize, &maxSupportSize, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF, maxSupportSize+1, &multiplicity);CHKERRQ(ierr);
	ierr = VecSet(multiplicity, 0);CHKERRQ(ierr);
	for (i = 0; i < localVertexSize; i++) {
		const PetscInt	point = vertexidx[i];
		PetscInt adjsize, *adj;
		ierr = DMPlexGetSupportSize(dm, point, &localSupportSize);CHKERRQ(ierr);
		ierr = DMPlexGetAdjacency(dmin, point, &adjsize, &adj);CHKERRQ(ierr);
		PetscPrintf(comm, "%d\n", adjsize);
		ierr = VecSetValue(multiplicity, localSupportSize, 1, ADD_VALUES);CHKERRQ(ierr);
		if (localSupportSize > locmaxMultiplicity) {
			locmaxMultiplicity = localSupportSize;
		}
		if (localSupportSize < locminMultiplicity) {
			locminMultiplicity = localSupportSize;
		}
	}
	ierr = ISRestoreIndices(vertexIS, &vertexidx);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmaxMultiplicity, &maxMultiplicity, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locminMultiplicity, &minMultiplicity, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
	PetscInt	globalmultiplicities[maxSupportSize+1];
	for (i = 0; i < maxSupportSize+1; i++) {
		globalmultiplicities[i] = 0.0;
	}
	ierr = VecGetArray(multiplicity, &locmultiplicities);CHKERRQ(ierr);
	ierr = MPIU_Allreduce(&locmultiplicities, &globalmultiplicities, maxSupportSize, MPIU_INT, MPI_SUM, comm);CHKERRQ(ierr);

	/* Nodes Per Process	*/
	ierr = VecCreateMPI(comm, 1, size, &nodesPerProcess);CHKERRQ(ierr);
	ierr = VecSet(nodesPerProcess, 0);CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingCreate(comm, 1, 1, &rank, PETSC_COPY_VALUES, &ltog);;CHKERRQ(ierr);
	ierr = VecSetLocalToGlobalMapping(nodesPerProcess, ltog);CHKERRQ(ierr);
	ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
	ierr = DMPlexGetDepthStratum(dmin, 0, &cstart, &cend);CHKERRQ(ierr);
	ierr = VecSetValueLocal(nodesPerProcess, 0, cend-cstart, INSERT_VALUES);CHKERRQ(ierr);
	ierr = VecUniqueEntries(nodesPerProcess, &numBinnedNodes, &binnedNodes);CHKERRQ(ierr);
	PetscInt	cellPerNodeCount[numBinnedNodes];
	for (i = 0; i < numBinnedNodes; i++) {
		ierr = VecTaggerCreate(comm, &tagger);CHKERRQ(ierr);
		ierr = VecTaggerSetType(tagger, VECTAGGERABSOLUTE);CHKERRQ(ierr);
		ierr = PetscMalloc1(1, &box);CHKERRQ(ierr);
		box->min = binnedNodes[i]-0.5;
		box->max = binnedNodes[i]+0.5;
		ierr = VecTaggerAbsoluteSetBox(tagger, box);CHKERRQ(ierr);
		ierr = PetscFree(box);CHKERRQ(ierr);
		ierr = VecTaggerSetFromOptions(tagger);CHKERRQ(ierr);
		ierr = VecTaggerSetUp(tagger);CHKERRQ(ierr);
		ierr = VecTaggerComputeIS(tagger, nodesPerProcess, &cellPerNodeTaggedIS);CHKERRQ(ierr);
		ierr = ISGetSize(cellPerNodeTaggedIS, &cellPerNodeCount[i]);CHKERRQ(ierr);
	}
	/* Printing	*/
	ierr = PetscViewerASCIIPrintf(viewer, "Distributed DM:%s>%s\n", bar, dmDistributed ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Interpolated DM:%s>%s\n", bar + 1, dmInterped ? "PETSC_TRUE *" : "PETSC_FALSE");CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Dimension of mesh:%s>%d\n", bar + 3, dim);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Global Vertex Num:%s>%d\n", bar + 3, globalVertexSize);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Global Edge Num:%s>%d\n", bar + 1, globalEdgeSize);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Global Face Num:%s>%d\n", bar + 1, globalFaceSize);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Global Cell Num:%s>%d\n\n", bar + 1, globalCellSize);CHKERRQ(ierr);
	ierr = PetscViewerASCIIPrintf(viewer, "Vertex Multiplicity Range: %d - %d\n", minMultiplicity, maxMultiplicity);CHKERRQ(ierr);
	for (i = minMultiplicity; i <= maxMultiplicity; i++) {
		if (i == minMultiplicity) {
			ierr = PetscViewerASCIIPrintf(viewer, "Multipl:NumVert\n");CHKERRQ(ierr);
		}
		ierr = PetscViewerASCIIPrintf(viewer, "\t%d - %d\n", i, globalmultiplicities[i]);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer, "\nElems Per Process Range: %.0f - %.0f\n", binnedNodes[0], binnedNodes[numBinnedNodes-1]);CHKERRQ(ierr);
	for (i = 0; i < numBinnedNodes; i++) {
		if (i == 0) {
			ierr = PetscViewerASCIIPrintf(viewer, "ElemsPP:Num Processes\n");CHKERRQ(ierr);
		}
		ierr = PetscViewerASCIIPrintf(viewer, "\t%.0f - %d\n", binnedNodes[i], cellPerNodeCount[i]);CHKERRQ(ierr);
	}
	ierr = PetscViewerASCIIPrintf(viewer, "%s End General Info %s\n", bar + 2, bar + 5);CHKERRQ(ierr);
	ierr = VecRestoreArray(multiplicity, &locmultiplicities);CHKERRQ(ierr);
	ierr = VecDestroy(&multiplicity);CHKERRQ(ierr);
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
	PetscInt		i, dim = 2, numFields = 1, numBC = 1, faces[dim], bcField[numBC];
	PetscBool		overlap = PETSC_FALSE, simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

	ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if(ierr){ return ierr;}
        comm = PETSC_COMM_WORLD;

	for (i = 0; i < dim; i++) {
		faces[i] = 2;
	}
	ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, dmInterped, &dm);CHKERRQ(ierr);
	ierr = DMPlexDistribute(dm, overlap, NULL, &dmDist);CHKERRQ(ierr);
	if (dmDist) {
		ierr = DMDestroy(&dm);CHKERRQ(ierr);
		dm = dmDist;
	}

	PetscInt        numDOF[numFields*(dim+1)], numComp[numFields];
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

	DMView(dm, 0);
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
