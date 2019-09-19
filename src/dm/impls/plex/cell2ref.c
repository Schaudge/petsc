# include <petscdmplex.h>
# include <petscviewer.h>
# include <petsc/private/dmpleximpl.h>

# define ANSI_RED "\033[1;31m"
# define ANSI_GREEN "\033[1;32m"
# define ANSI_RESET "\033[0m"

PetscErrorCode OrthoganalQuality(MPI_Comm comm, DM dm, Vec *OrthQual)
{
  PetscErrorCode	ierr;
  IS			cellIS, subAlloc;
  const PetscInt	*cells;
  PetscInt		cStart, cEnd, cellIter, nPointsPerCell, i;
  Vec			temp, cent2faces, faceNormVec;

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
    PetscScalar		OrthQualPerFace = 0.0, OrthQualPerCell = 0.0, Anorm, Fnorm, DotProd= 0.0;
    ierr = VecZeroEntries(cent2faces);CHKERRQ(ierr);
    ierr = VecZeroEntries(faceNormVec);CHKERRQ(ierr);
    ierr = CentroidToFace(dm, cell, nPointsPerCell, &cent2faces, &faceNormVec);CHKERRQ(ierr);
    for (i = 0; i < nPointsPerCell; i++) {
      PetscInt		*idx;
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

PetscErrorCode CentroidToFace(DM dm, const PetscInt cellid, PetscInt nPointsPerCell, Vec *cent2faces, Vec *faceNormVec)
{
  PetscErrorCode	ierr;
  PetscSection 		cSection;
  Vec 			cellCoord;
  const PetscInt	*faces;
  PetscInt		p, offset, minOff;
  PetscInt		points[nPointsPerCell];
  PetscScalar		xsum = 0.0, ysum = 0.0;
  PetscScalar		*cArray, *c2farr, centCoord[2], faceCent[2];

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
    PetscInt		i;
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

PetscErrorCode Cell2Coords(DM dm, PetscInt cellId, PetscInt *points)
{
  PetscErrorCode	ierr;
  PetscInt		zeroiter, edgeIter, numConnEdges;
  const PetscInt	*connEdges;

  ierr = DMPlexGetCone(dm, cellId, &connEdges);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cellId, &numConnEdges);CHKERRQ(ierr);

  for (zeroiter = 0; zeroiter < numConnEdges; zeroiter++) {
    points[zeroiter] = -1;
  }
  for (edgeIter = 0;edgeIter < numConnEdges; edgeIter++) {
    PetscInt		pointIter, numConnPoints;
    const PetscInt	*connPoints;

    ierr = DMPlexGetCone(dm, connEdges[edgeIter], &connPoints);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, connEdges[edgeIter], &numConnPoints);CHKERRQ(ierr);
    for (pointIter = 0; pointIter < numConnPoints; pointIter++) {
      PetscBool		inArray = PETSC_FALSE;

      valueInArray(connPoints[pointIter], points, numConnEdges, &inArray);
      if (!inArray) {
        points[edgeIter+pointIter] = connPoints[pointIter];
      }
    }
  }
  return ierr;
}

PetscErrorCode FaceNormPerCell(DM dm, PetscSection cSection, const PetscInt faceid[], PetscInt idx,  Vec *faceNormVec)
{
  PetscErrorCode	ierr;
  Vec			coords;
  PetscInt		offset0, offset1, minOff;
  PetscScalar		dx = 0.0, dy = 0.0;
  PetscScalar		*cArray, *fArray;

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

PetscErrorCode valueInArray(const PetscInt val, PetscInt *arr, PetscInt sizeOfArr, PetscBool *inArray)
{
  PetscInt	i;
  for(i = 0; i < sizeOfArr; i++) {
    if(arr[i] == val) {
      *inArray = PETSC_TRUE;
      return 1;
    }
  }
  *inArray = PETSC_FALSE;
  return (0);
}

PetscErrorCode StretchArray2D(DM dm, PetscScalar lx, PetscScalar ly)
{
  PetscErrorCode          ierr;
  PetscInt                i, nCoords;
  Vec                     coordsLocal;
  PetscScalar             *coordArray;

  ierr = DMGetCoordinates(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(coordsLocal, &nCoords);CHKERRQ(ierr);
  ierr = VecGetArray(coordsLocal, &coordArray);CHKERRQ(ierr);

  // Order in coordarray is [x1,y1,z1....]
  for (i = 0; i < nCoords; i++) {
    //if ((i < 6) || (i > 11)) {
    if (i % 2) {
      coordArray[i-1] = lx*coordArray[i-1];
      coordArray[i] = ly*coordArray[i];
    }
    // }
  }
  ierr = VecRestoreArray(coordsLocal, &coordArray);CHKERRQ(ierr);
  ierr = DMSetCoordinates(dm, coordsLocal);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode SkewArray2D(DM dm, PetscScalar omega)
{
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

PetscErrorCode Matvis(const char prefix[], PetscScalar mat[])
{
  PetscErrorCode	ierr;

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s ->\t[%2.2f, %2.2f, %2.2f]\n\t\t[%2.2f, %2.2f, %2.2f]\n\t\t[%2.2f, %2.2f, %2.2f]\n", prefix, mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8]);CHKERRQ(ierr);
  return ierr;
}

PetscErrorCode AngleBetweenConnectedEdges(DM dm, PetscInt *foundcells, PetscInt numCells, PetscInt vertex, PetscScalar *angles[], PetscInt *startEdge)
{
  PetscErrorCode	ierr;
  const PetscInt	*edges, *vertsOnEdge;
  PetscInt		i, j, numEdges, numVerts, dim, vStart, refVert, compVert;
  PetscScalar		refx, refy, compx, compy, centerx, centery, det, dot, x;
  PetscScalar		*carr, *angles_;
  Vec			coordinates;

  printf("--------------------- ANGLES --------------------");
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, NULL);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,  &dim);CHKERRQ(ierr);
  ierr = DMPlexGetSupport(dm, vertex, &edges);CHKERRQ(ierr);
  ierr = DMPlexGetSupportSize(dm, vertex, &numEdges);CHKERRQ(ierr);
  printf("\nNUMBER OF EDGES: %2d\n", numEdges);
  ierr = PetscCalloc1(numEdges, &angles_);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, edges[0], &vertsOnEdge);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, edges[0], &numVerts);CHKERRQ(ierr);
  for (i = 0; i < numVerts; i++) {
    if (vertsOnEdge[i] != vertex) { refVert = vertsOnEdge[i];}
  }
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &carr);CHKERRQ(ierr);
  centerx = carr[dim*(vertex-vStart)]; centery = carr[dim*(vertex-vStart)+1];
  refx = carr[dim*(refVert-vStart)]-centerx; refy = carr[dim*(refVert-vStart)+1]-centery;
  printf("REFERENCE VERTEX: %d -> (%2.2f,%2.2f)\n\n", refVert, refx, refy);
  for (i = 1; i < numEdges; i++) {
    printf("EDGE: %2d\n", edges[i]);
    ierr = DMPlexGetCone(dm, edges[i], &vertsOnEdge);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, edges[i], &numVerts);CHKERRQ(ierr);
    for (j = 0; j < numVerts; j++) {
      //printf("CURRENT %2d --- COMPARE %2d\n", vertsOnEdge[j], vertex);
      if (vertsOnEdge[j] != vertex) { compVert = vertsOnEdge[j];}
    }
    compx = carr[dim*(compVert-vStart)]-centerx; compy = carr[dim*(compVert-vStart)+1]-centery;
    printf("Chosen Vertex:\t  %2.d -> (%2.2f,%2.2f)\n", compVert, compx, compy);
    dot = (refx*compx) + (refy*compy);
    det = (refx*compy) - (refy*compx);
    printf("DOT: %2.2f\nDET: %2.2f\n", dot, det);
    x = PetscAtan2Real(det, dot);
    angles_[i-1] = (x > 0 ? x : (2*PETSC_PI + x)) * 360 / (2*PETSC_PI);
    printf("COMPUTED ANGLE: %f\n", angles_[i-1]);
  }
  ierr = VecRestoreArray(coordinates, &carr);CHKERRQ(ierr);
  ierr = PetscSortReal(numEdges, angles_);CHKERRQ(ierr);
  for (i = 0; i < numEdges-1; i++) {
    angles_[i] = angles_[i+1]-angles_[i];
  }
  angles_[numEdges-1] = 360-angles_[numEdges-2];
  ierr = PetscArraycpy(*angles, angles_, numEdges);CHKERRQ(ierr);
  *startEdge = edges[0];
  ierr = PetscFree(angles_);CHKERRQ(ierr);
  printf("-------------------------------------------------");
  return ierr;
}

PetscErrorCode RemoveDupsArray(const PetscInt unsortarr[], PetscInt noduparr[], PetscInt ntotal, PetscInt n, PetscInt search, PetscInt *loc)
{
  PetscInt	i, j, k = 0;


  for (i = 0; i < ntotal; i++) {
    PetscInt 	key = unsortarr[i];
    PetscBool	found = PETSC_FALSE;
    for (j = 0; j < n; j++) {
      if (noduparr[j] == key) {
        found = PETSC_TRUE;
      }
    }
    if (!found) {
      noduparr[k] = key;
      if (key == search) { *loc = k;}
      k++;
    }
  }
  return (0);
}

PetscErrorCode ComputeR2X2RMapping(DM dm, PetscInt vertex, PetscInt cell, PetscScalar R2Xmat[], PetscScalar X2Rmat[], PetscScalar realC_[], PetscScalar refC_[])
{
  PetscErrorCode	ierr;
  IS      		singleCellIS, vertsIS, vertsISfake;
  Vec			coords;
  PetscInt		idx[1] = {cell}, *nodupidx;
  PetscInt		dim, i, nverts, ntotal, vStart, loc = 0, tempi, tempi2, tempi3;
  const PetscInt	*ptr;
  PetscScalar		*xtilde, *rtilde, *invR, *coordArray;
  PetscScalar		detR2X, detR;
  PetscBool		USE_ROTATION = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL, NULL, "-rot", &USE_ROTATION, NULL);CHKERRQ(ierr);
  printf("USING ROTATION:\t%s%s%s\n", USE_ROTATION ? ANSI_GREEN : ANSI_RED , USE_ROTATION ? "PETSC_TRUE" : "PETSC_FALSE", ANSI_RESET);

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  dim = dim+1;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &xtilde);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &rtilde);CHKERRQ(ierr);
  rtilde[0] = 0.0; rtilde[1] = 0.0; rtilde[2] = 1.0;
  rtilde[3] = 0.0; rtilde[4] = 1.0; rtilde[5] = 1.0;
  rtilde[6] = 1.0; rtilde[7] = 1.0; rtilde[8] = 1.0;
  xtilde[6] = 1.0; xtilde[7] = 1.0; xtilde[8] = 1.0;

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, idx, PETSC_COPY_VALUES, &singleCellIS);CHKERRQ(ierr);
  ierr = DMPlexGetConeRecursiveVertices(dm, singleCellIS, &vertsIS);CHKERRQ(ierr);
  ierr = ISDuplicate(vertsIS, &vertsISfake);CHKERRQ(ierr);
  ierr = ISSortRemoveDups(vertsISfake);CHKERRQ(ierr);
  ierr = ISGetSize(vertsISfake, &nverts);CHKERRQ(ierr);
  ierr = ISGetSize(vertsIS, &ntotal);CHKERRQ(ierr);
  ierr = ISDestroy(&vertsISfake);CHKERRQ(ierr);
  ierr = PetscCalloc1(nverts, &nodupidx);CHKERRQ(ierr);
  ierr = ISGetIndices(vertsIS, &ptr);CHKERRQ(ierr);
  ierr = RemoveDupsArray(ptr, nodupidx, ntotal, nverts, vertex, &loc);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vertsIS, &ptr);CHKERRQ(ierr);
  printf("LOC: %d\n", loc);
  PetscIntView(nverts, nodupidx, 0);
  for (i = nverts-1; i > 0; i--) {
    PetscScalar	xval, yval, detX;
    PetscBool	SUCESS = PETSC_FALSE;

    tempi = (loc+i+1)%nverts;
    if (tempi-1 < 0) 	{ tempi2 = nverts-1;} else { tempi2 = tempi-1;}
    if (tempi2-1 < 0) 	{ tempi3 = nverts-1;} else { tempi3 = tempi2-1;}
    xval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    yval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];

    printf("CURRENT %d\t -> [%.1f %.1f]\nNEXT \t%d\t -> [%.1f %.1f]\nNEXT \t%d\t -> [%.1f %.1f]\n", nodupidx[tempi], xval, yval, nodupidx[tempi2], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1], nodupidx[tempi3], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1]);

    xtilde[0] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    xtilde[1] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)];
    xtilde[2] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)];
    xtilde[3] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];
    xtilde[4] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1];
    xtilde[5] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1];
    printf("But wait! Theres more! Check DETERMINANT\n");
    DMPlex_Det3D_Internal(&detX, xtilde);
    printf("%sDETX%s:\t\t%f\n", (PetscAbs(detX) > 0) ? ANSI_GREEN : ANSI_RED, ANSI_RESET, PetscAbs(detX));
    if (PetscAbs(detX) > 0) {
      printf("USING:\t\t%d %d %d\n\n", nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3]);
      SUCESS = PETSC_TRUE;
    } else {
      printf("%sZERO DETERMINANT%s: %d %d %d -> %.1f\n", ANSI_RED, ANSI_RESET, nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3], detX);
      i--;
    }
    if (SUCESS) { i = 0;} else {
      printf("%sNO SUITABLE TRANSFORM FOR CELL%s: %d\n", ANSI_RED, ANSI_RESET, cell);
      return (0);
    }
  }
  ierr = PetscCalloc1(dim*dim, &invR);CHKERRQ(ierr);
  DMPlex_Det3D_Internal(&detR, rtilde);
  DMPlex_Invert3D_Internal(invR, rtilde, detR);
  DMPlex_MatMult3D_Internal(xtilde, dim, dim, invR, R2Xmat);
  ierr = Matvis("XTmat", xtilde);CHKERRQ(ierr);
  ierr = Matvis("RTmat", rtilde);CHKERRQ(ierr);
  DMPlex_Det3D_Internal(&detR2X, R2Xmat);
  DMPlex_Invert3D_Internal(X2Rmat, R2Xmat, detR2X);

  printf("\n");
  for (i = 0; i < nverts; i++) {
    PetscScalar x, y;
    PetscScalar	*realC, *refC;

    ierr = PetscCalloc1(dim, &refC);CHKERRQ(ierr);
    ierr = PetscCalloc1(dim,&realC);CHKERRQ(ierr);
    x = coordArray[(dim-1)*(nodupidx[i]-vStart)];
    y = coordArray[(dim-1)*(nodupidx[i]-vStart)+1];
    realC[0] = x; realC[1] = y; realC[2] = 1.0;

    DMPlex_Mult3D_Internal(X2Rmat, 1, realC, refC);
    if (nodupidx[i] == vertex) { printf("++++++++++++++++++++++++++++++++++++++++++++++++\n");}
    printf("FOR CELL %3d, VERTEX %3d REALC: (%.3f, %.3f) -> REFC: (%.3f, %.3f)\n", cell, nodupidx[i], realC[0], realC[1], refC[0], refC[1]);

    if ((nodupidx[i] == vertex) && USE_ROTATION) {
      PetscScalar	xc = 0.5, yc = 0.5, theta;
      PetscScalar	*rotMat, *X2Rtemp;
      PetscInt		k;

      ierr = PetscCalloc1(dim*dim, &X2Rtemp);CHKERRQ(ierr);
      for (k = 0; k < dim*dim; k++) {
        X2Rtemp[k] = X2Rmat[k];
      }
      ierr = PetscCalloc1(dim*dim, &rotMat);CHKERRQ(ierr);
      rotMat[0] = 1; rotMat[4] = 1; rotMat[8] = 1;

      if ((PetscAbs(refC[0]) > 0.1) || (PetscAbs(refC[1]) > 0.1)) {
        printf("%f %f\n", refC[0], refC[1]);
        if (refC[0] == refC[1]) { theta = PETSC_PI;} else { theta = refC[1] > refC[0] ? PETSC_PI/2 : -1.0*PETSC_PI/2;}
        rotMat[0] = PetscCosReal(theta); rotMat[1] = -1.0*PetscSinReal(theta);
        rotMat[2] = (-xc*PetscCosReal(theta)) + (yc*PetscSinReal(theta)) + xc;
        rotMat[3] = PetscSinReal(theta); rotMat[4] = PetscCosReal(theta);
        rotMat[5] = (-xc*PetscSinReal(theta)) - (yc*PetscCosReal(theta)) + yc;
        DMPlex_MatMult3D_Internal(rotMat, dim, dim, X2Rmat, X2Rtemp);
        for (k = 0; k < dim*dim; k++) {
          X2Rmat[k] = X2Rtemp[k];
        }
        ierr = Matvis("X2R + ROT", X2Rmat);CHKERRQ(ierr);
        DMPlex_Mult3D_Internal(X2Rmat, 1, realC, refC);
        printf("%f, %f, %f\n", theta, refC[0], refC[1]);
        i = -1;
      }
      ierr = PetscFree(rotMat);CHKERRQ(ierr);
      ierr = PetscFree(X2Rtemp);CHKERRQ(ierr);
    }
    if (nodupidx[i] == vertex) { printf("++++++++++++++++++++++++++++++++++++++++++++++++\n");}

    realC_[(dim-1)*i] = realC[0];
    realC_[((dim-1)*i)+1] = realC[1];
    refC_[(dim-1)*i] = refC[0];
    refC_[((dim-1)*i)+1] = refC[1];
    ierr = PetscFree(realC);CHKERRQ(ierr);
    ierr = PetscFree(refC);CHKERRQ(ierr);
  }
  printf("\n");
  DMPlex_Det3D_Internal(&detR2X, R2Xmat);
  DMPlex_Invert3D_Internal(X2Rmat, R2Xmat, detR2X);
  ierr = Matvis("R2Xmat", R2Xmat);CHKERRQ(ierr);
  ierr = Matvis("X2Rmat", X2Rmat);CHKERRQ(ierr);
  ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = ISDestroy(&vertsIS);CHKERRQ(ierr);
  ierr = ISDestroy(&singleCellIS);CHKERRQ(ierr);
  ierr = PetscFree(invR);CHKERRQ(ierr);
  ierr = PetscFree(nodupidx);CHKERRQ(ierr);
  ierr = PetscFree(xtilde);CHKERRQ(ierr);
  ierr = PetscFree(rtilde);CHKERRQ(ierr);
  return ierr;
}

int main(int argc, char **argv)
{
  MPI_Comm              comm;
  PetscErrorCode        ierr;
  PetscViewer		viewer, textviewer, hdf5viewer;
  DM                    dm, dmDist;
  IS                    bcPointsIS, globalCellIS, vertexIS;
  Vec			coords, OrthQual;
  PetscSection          section;
  PetscInt              overlap = 0, i, dim = 2, conesize, numFields = 1, numBC = 1, size, vsize, cEnd;
  PetscInt		faces[dim], bcField[numBC];
  const PetscInt	*ptr, *vptr;
  PetscScalar		*coordArray, *angles;
  PetscBool             simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm, "mesh.vtk", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &textviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(textviewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(textviewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(textviewer, "Angles.txt");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(textviewer);CHKERRQ(ierr);

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

  ierr = StretchArray2D(dm, 2.0, 1.0);CHKERRQ(ierr);
  ierr = SkewArray2D(dm, 45.0);CHKERRQ(ierr);

  ierr = DMPlexGetCellNumbering(dm, &globalCellIS);CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "depth", 0, &vertexIS);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &cEnd);CHKERRQ(ierr);
  ierr = ISGetIndices(globalCellIS, &ptr);CHKERRQ(ierr);
  ierr = ISGetIndices(vertexIS, &vptr);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, ptr[0], &conesize);CHKERRQ(ierr);
  ierr = ISGetSize(globalCellIS, &size);CHKERRQ(ierr);
  ierr = ISGetSize(vertexIS, &vsize);CHKERRQ(ierr);

  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);
  for (i = 0; i < vsize; i++) {
    PetscInt	vertex = vptr[i];
    PetscInt	*points, *foundcells;
    PetscInt	numPoints, numEdges, j, actualj, cell, k = 0, sEdge;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
    printf("VERTEX# : %d -> (%.3f , %.3f) ", vertex, coordArray[2*i], coordArray[2*i+1]);
    ierr = PetscCalloc1(conesize, &foundcells);CHKERRQ(ierr);
    for (j = 0; j < numPoints; j++) {
      actualj = 2*j;
      cell = points[actualj];
      if (cell < cEnd) {
        foundcells[k] = cell;
        k++;
      }
    }
    printf("For Vertex %d found %d cells\n", vertex, k);
    ierr = DMPlexGetSupportSize(dm, vertex, &numEdges);CHKERRQ(ierr);
    ierr = PetscCalloc1(numEdges, &angles);CHKERRQ(ierr);
    ierr = AngleBetweenConnectedEdges(dm, foundcells, k, vertex, &angles, &sEdge);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(textviewer, "#NUMEDGE %d VERT %d StartEdge %d\n", numEdges, vertex, sEdge);CHKERRQ(ierr);
    for (j = 0; j < numEdges; j++) {
      ierr = PetscViewerASCIIPrintf(textviewer, "%f\n", angles[j]);CHKERRQ(ierr);
    }
    ierr = PetscFree(angles);CHKERRQ(ierr);
    for (j = 0; j < k; j++) {
      PetscScalar	*R2Xmat, *X2Rmat, *realCtemp, *refCtemp;

      ierr = PetscCalloc1((dim+1)*(dim+1), &R2Xmat);CHKERRQ(ierr);
      ierr = PetscCalloc1((dim+1)*(dim+1), &X2Rmat);CHKERRQ(ierr);
      ierr = PetscCalloc1((dim+1)*conesize, &realCtemp);CHKERRQ(ierr);
      ierr = PetscCalloc1((dim+1)*conesize, &refCtemp);CHKERRQ(ierr);
      printf("\ncell: %d, vertex: %d\n", foundcells[j], vertex);
      ierr = ComputeR2X2RMapping(dm, vertex, foundcells[j], R2Xmat, X2Rmat, realCtemp, refCtemp);CHKERRQ(ierr);

      ierr = PetscFree(R2Xmat);CHKERRQ(ierr);
      ierr = PetscFree(X2Rmat);CHKERRQ(ierr);
      ierr = PetscFree(realCtemp);CHKERRQ(ierr);
      ierr = PetscFree(refCtemp);CHKERRQ(ierr);
    }
    printf("=====================================================\n");
    ierr = PetscFree(foundcells);CHKERRQ(ierr);
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = ISRestoreIndices(globalCellIS, &ptr);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vertexIS, &vptr);CHKERRQ(ierr);
  ierr = ISDestroy(&vertexIS);CHKERRQ(ierr);

  ierr = VecCreateSeq(comm, cEnd, &OrthQual);CHKERRQ(ierr);
  ierr = OrthoganalQuality(comm, dm, &OrthQual);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &textviewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(textviewer, PETSCVIEWERASCII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(textviewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(textviewer, "Orthqual.txt");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(textviewer);CHKERRQ(ierr);
  ierr = VecView(OrthQual, textviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&textviewer);CHKERRQ(ierr);
  ierr = VecDestroy(&OrthQual);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)coords, "Deformed");CHKERRQ(ierr);
  ierr = DMPlexVTKWriteAll((PetscObject) dm, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm, &hdf5viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(hdf5viewer, PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(hdf5viewer, FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(hdf5viewer, "Mesh.H5");CHKERRQ(ierr);
  ierr = PetscViewerSetUp(hdf5viewer);CHKERRQ(ierr);
  ierr = DMView(dm, hdf5viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf5viewer);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
