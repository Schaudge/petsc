static char help[33] = "Concise View of DMPlex Object\n";

# include <petscdmplex.h>

PetscErrorCode DMPlexComputeCellOrthogonalQuality(DM dm, Vec *OrthogonalQuality)
{
  MPI_Comm              comm;
  PetscObject    	cellgeomobj, facegeomobj;
  PetscErrorCode        ierr;
  IS			centIS, fcentIS, fnormIS, subCellIS, subFaceIS;
  Vec                   cellGeom, faceGeom, subCell, subFace, subCellCent, subFaceCent, subFaceNormal;
  PetscInt		celliter, faceiter, i, j, cellHeight, dim, depth, cStart, cEnd, fStart, numFaces;
  PetscInt		*cdx, *fdx, *centdx, *fcentdx, *fnormdx;
  PetscScalar		*OrthogonalQualityArray;
  size_t		subCellVecSize = 4, subFaceVecSize = 12, centVecSize = 3, normalVecSize = 3;

  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_cellgeom_fvm", &cellgeomobj);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "DMPlex_facegeom_fvm", &facegeomobj);CHKERRQ(ierr);
  if ((!cellgeomobj) || (!facegeomobj)) {
      ierr = DMPlexComputeGeometryFVM(dm, &cellGeom, &faceGeom);CHKERRQ(ierr);
  } else {
    cellGeom = (Vec) cellgeomobj;
    faceGeom = (Vec) facegeomobj;
  }
  VecView(cellGeom, 0);
  VecView(faceGeom, 0);
  ierr = DMPlexGetVTKCellHeight(dm, &cellHeight);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, cellHeight+1, &fStart, NULL);CHKERRQ(ierr);
  if (!*OrthogonalQuality) {
    ierr = DMCreateGlobalVector(dm, OrthogonalQuality);CHKERRQ(ierr);
    ierr = VecZeroEntries(*OrthogonalQuality);CHKERRQ(ierr);
  }
  ierr = VecGetArray(*OrthogonalQuality, &OrthogonalQualityArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(subCellVecSize, &cdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(subFaceVecSize, &fdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(centVecSize, &centdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(centVecSize, &fcentdx);CHKERRQ(ierr);
  ierr = PetscMalloc1(normalVecSize, &fnormdx);CHKERRQ(ierr);
  centdx[0] = 0; centdx[1] = 1; centdx[2] = 2;
  fcentdx[0] = 3; fcentdx[1] = 4; fcentdx[2] = 5;
  fnormdx[0] = 0; fnormdx[1] = 1; fnormdx[2] = 2;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, centVecSize, centdx, PETSC_COPY_VALUES, &centIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, centVecSize, fcentdx, PETSC_COPY_VALUES, &fcentIS);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, normalVecSize, fnormdx, PETSC_COPY_VALUES, &fnormIS);CHKERRQ(ierr);

  ierr = PetscFree(centdx);CHKERRQ(ierr);
  ierr = PetscFree(fcentdx);CHKERRQ(ierr);
  ierr = PetscFree(fnormdx);CHKERRQ(ierr);
  for (celliter = cStart; celliter < cEnd; celliter++) {
    PetscScalar		OrthQualPerFace = 0.0, OrthQualPerCell = 0.0, Anorm, Fnorm, DotProd = 0.0;
    PetscInt		*coneCopy;
    const PetscInt	*cone;

    ierr = DMPlexGetConeSize(dm, celliter, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, celliter, &cone);CHKERRQ(ierr);
    ierr = PetscMalloc1(numFaces, &coneCopy);CHKERRQ(ierr);
    for (i = 0; i < numFaces; i++) {
      printf("CONE: %d\n", cone[i]);
      coneCopy[i] = cone[i] - fStart;
    }
    for (j = 0; j < subCellVecSize; j++) {
      cdx[j] = (subCellVecSize*celliter)+j;
    }
    PetscPrintf(comm, "=========================== ");
    PetscPrintf(comm, "cell #%d\n", celliter);
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, subCellVecSize, cdx, PETSC_COPY_VALUES, &subCellIS);CHKERRQ(ierr);
    ierr = VecGetSubVector(cellGeom, subCellIS, &subCell);CHKERRQ(ierr);
    ierr = VecGetSubVector(subCell, centIS, &subCellCent);CHKERRQ(ierr);
    //VecView(subCell,0);
    for (faceiter = 0; faceiter < numFaces; faceiter++) {
      Vec		cent2face;
      PetscPrintf(comm, "face #%d\n", faceiter);
      PetscPrintf(comm, "sub alloc start: %d\n", subFaceVecSize*(coneCopy[faceiter]));

      for (j = 0; j < subFaceVecSize; j++) {
        fdx[j] = (subFaceVecSize*(coneCopy[faceiter]))+j;
      }

      ierr = ISCreateGeneral(PETSC_COMM_WORLD, subFaceVecSize, fdx, PETSC_COPY_VALUES, &subFaceIS);CHKERRQ(ierr);
      ierr = VecGetSubVector(faceGeom, subFaceIS, &subFace);CHKERRQ(ierr);
      ierr = VecGetSubVector(subFace, fcentIS, &subFaceCent);CHKERRQ(ierr);
      ierr = VecGetSubVector(subFace, fnormIS, &subFaceNormal);CHKERRQ(ierr);
      ierr = VecDuplicate(subFaceCent, &cent2face);CHKERRQ(ierr);
      ierr = VecWAXPY(cent2face, -1.0, subCellCent, subFaceCent);CHKERRQ(ierr);
      ierr = VecDot(cent2face, subFaceNormal, &DotProd);CHKERRQ(ierr);
      ierr = VecNorm(subFaceNormal, NORM_2, &Anorm);CHKERRQ(ierr);
      ierr = VecNorm(cent2face, NORM_2, &Fnorm);CHKERRQ(ierr);

      ierr = VecDestroy(&cent2face);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(subFace, fnormIS, &subFaceNormal);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(subFace, fcentIS, &subFaceCent);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(faceGeom, subFaceIS, &subFace);CHKERRQ(ierr);
      ierr = ISDestroy(&subFaceIS);CHKERRQ(ierr);
    }
    ierr = PetscFree(coneCopy);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(subCell, centIS, &subCellCent);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(cellGeom, subCellIS, &subCell);CHKERRQ(ierr);
    ierr = ISDestroy(&subCellIS);CHKERRQ(ierr);
  }
  ierr = PetscFree(cdx);CHKERRQ(ierr);
  ierr = PetscFree(fdx);CHKERRQ(ierr);
  return ierr;
}

int main(int argc, char **argv)
{
  MPI_Comm              comm;
  PetscErrorCode        ierr;
  IS                    bcPointsIS;
  PetscSection          section;
  Vec			OrthogonalQuality;
  DM                    dm, dmDist;
  PetscInt              overlap = 0, i, dim = 2, numFields = 1, numBC = 1, faces[dim], bcField[numBC];
  PetscBool             simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

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
  PetscInt      numDOF[numFields*(dim+1)], numComp[numFields];
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

  ierr = DMPlexComputeCellOrthogonalQuality(dm, &OrthogonalQuality);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
