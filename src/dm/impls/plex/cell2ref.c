# include <petscdmplex.h>
# include <petscviewer.h>
#include <petsc/private/dmpleximpl.h>

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

PetscErrorCode RemoveDupsArray(const PetscInt unsortarr[], PetscInt noduparr[], PetscInt ntotal, PetscInt n, PetscInt search, PetscInt loc)
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
      if (key == search) { loc = k;}
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
  ierr = RemoveDupsArray(ptr, nodupidx, ntotal, nverts, vertex, loc);CHKERRQ(ierr);
  ierr = ISRestoreIndices(vertsIS, &ptr);CHKERRQ(ierr);
  printf("LOC: %d\n", loc);
  PetscIntView(nverts, nodupidx, 0);
  for (i = nverts-1; i > 0; i--) {
    PetscScalar	xval, yval, detX;

    tempi = (loc+i+1)%nverts;
    if (tempi-1 < 0) 	{ tempi2 = nverts-1;} else { tempi2 = tempi-1;}
    if (tempi2-1 < 0) 	{ tempi3 = nverts-1;} else { tempi3 = tempi2-1;}
    xval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    yval = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];

    printf("CURRENT %d\t -> [%.1f %.1f]\nNEXT %d\t\t -> [%.1f %.1f]\nNEXT %d\t\t -> [%.1f %.1f]\n", nodupidx[tempi], xval, yval, nodupidx[tempi2], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)], coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1], nodupidx[tempi3], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)], coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1]);

    xtilde[0] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)];
    xtilde[1] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)];
    xtilde[2] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)];
    xtilde[3] = coordArray[(dim-1)*(nodupidx[tempi]-vStart)+1];
    xtilde[4] = coordArray[(dim-1)*(nodupidx[tempi2]-vStart)+1];
    xtilde[5] = coordArray[(dim-1)*(nodupidx[tempi3]-vStart)+1];
    printf("But wait! Theres more! Check DETERMINANT\n");
    DMPlex_Det3D_Internal(&detX, xtilde);
    printf("DETX %f\n", PetscAbs(detX));
    if (PetscAbs(detX) > 0) {
      printf("USING:\t\t %d, %d %d\n\n", nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3]);
      i = 0;
    } else {
      printf("%d, %d %d ZERO DETERMINANT: %.1f\n", nodupidx[tempi], nodupidx[tempi2], nodupidx[tempi3], detX);
      i--;
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
  PetscViewer		viewer;
  DM                    dm, dmDist;
  IS                    bcPointsIS, globalCellIS, vertexIS;
  Vec			coords;
  PetscSection          section;
  PetscInt              overlap = 0, i, dim = 2, conesize, numFields = 1, numBC = 1, size, vsize, cEnd;
  PetscInt		faces[dim], bcField[numBC];
  const PetscInt	*ptr, *vptr;
  PetscScalar		*coordArray;
  PetscBool             simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerVTKOpen(comm, "mesh.vtk", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);

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
    PetscInt	numPoints, j, actualj, cell, k = 0;

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

  ierr = PetscObjectSetName((PetscObject)coords, "Deformed");CHKERRQ(ierr);
  ierr = DMPlexVTKWriteAll((PetscObject) dm, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
