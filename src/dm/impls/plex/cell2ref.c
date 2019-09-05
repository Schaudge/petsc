# include <petscdmplex.h>
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
            //}
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

PetscErrorCode ComputeR2X2RMapping(DM dm, PetscInt vertex, PetscInt cell, PetscScalar R2Xmat[], PetscScalar X2Rmat[])
{
  PetscErrorCode	ierr;
  IS      		singleCellIS, vertsIS, vertsISfake;
  Vec			coords;
  PetscInt		idx[1] = {cell}, *nodupidx;
  PetscInt		dim, i, k = 0, nverts, ntotal, vStart;
  const PetscInt	*ptr;
  PetscScalar		*xtilde, *rtilde, *invR, *coordArray;
  PetscScalar		detR2X, detR;

  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, NULL);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coords, &coordArray);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &xtilde);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim*dim, &rtilde);CHKERRQ(ierr);
  rtilde[0] = 0.0; rtilde[1] = 1.0; rtilde[2] = 1.0; rtilde[3] = 1.0;

  ierr = ISCreateGeneral(PETSC_COMM_WORLD, 1, idx, PETSC_COPY_VALUES, &singleCellIS);CHKERRQ(ierr);
  ierr = DMPlexGetConeRecursiveVertices(dm, singleCellIS, &vertsIS);CHKERRQ(ierr);
  ierr = ISDuplicate(vertsIS, &vertsISfake);CHKERRQ(ierr);
  ierr = ISSortRemoveDups(vertsISfake);CHKERRQ(ierr);
  ierr = ISGetSize(vertsISfake, &nverts);CHKERRQ(ierr);
  ierr = ISGetSize(vertsIS, &ntotal);CHKERRQ(ierr);
  ierr = ISDestroy(&vertsISfake);CHKERRQ(ierr);
  ierr = PetscCalloc1(nverts, &nodupidx);CHKERRQ(ierr);
  ierr = ISGetIndices(vertsIS, &ptr);CHKERRQ(ierr);
  for (i = 0; i < ntotal; i++) {
    PetscInt 	key = ptr[i], j;
    PetscBool	found = PETSC_FALSE;
    for (j = 0; j < nverts; j++) {
      if (nodupidx[j] == key) {
        found = PETSC_TRUE;
      }
    }
    if (!found) {
      nodupidx[k] = key;
      k++;
    }
  }
  ierr = ISRestoreIndices(vertsIS, &ptr);CHKERRQ(ierr);

  for (i = nverts-1; i > 0; i--) {
    printf("%d\n", i);
    PetscScalar	xval, yval;
    xval = coordArray[dim*(nodupidx[i]-vStart)];
    yval = coordArray[dim*(nodupidx[i]-vStart)+1];
    if (!((xval == 0.0) && (yval == 0.0))) {
      xval = coordArray[dim*(nodupidx[i-1]-vStart)];
      yval = coordArray[dim*(nodupidx[i-1]-vStart)+1];
      if (!((xval == 0.0) && (yval == 0.0))) {
        xtilde[0] = coordArray[dim*(nodupidx[i]-vStart)];
        xtilde[2] = coordArray[dim*(nodupidx[i]-vStart)+1];
        xtilde[1] = coordArray[dim*(nodupidx[i-1]-vStart)];
        xtilde[3] = coordArray[dim*(nodupidx[i-1]-vStart)+1];
        printf("FOUND: %f %f %f %f\n", xtilde[0], xtilde[1], xtilde[2], xtilde[3]);
        i = 0;
      } else {
        i--;
      }
    }
  }

  ierr = PetscCalloc1(nverts, &invR);CHKERRQ(ierr);
  DMPlex_Det2D_Internal(&detR, rtilde);
  printf("%f\n", detR);
  DMPlex_Invert2D_Internal(invR, rtilde, detR);
  DMPlex_MatMult2D_Internal(xtilde, dim, dim, invR, R2Xmat);
  DMPlex_Det2D_Internal(&detR2X, R2Xmat);
  DMPlex_Invert2D_Internal(X2Rmat, R2Xmat, detR2X);
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
  DM                    dm, dmDist;
  IS                    bcPointsIS, globalCellIS, vertexIS;
  Vec			coords, refCoords, cellGeom, faceGeom;
  PetscSection          section;
  PetscInt              overlap = 0, i, dim = 2, conesize, numFields = 1, numBC = 1, size, vsize, cEnd;
  PetscInt		faces[dim], bcField[numBC];
  const PetscInt	*ptr, *vptr;
  PetscScalar		*coordArray, refArray[8] = {0, 0, 1, 0, 0, 1, 1, 1}, *BIGX2Rmat, *BIGR2Xmat;
  PetscBool             simplex = PETSC_FALSE, dmInterped = PETSC_TRUE;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
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

  ierr = StretchArray2D(dm, 2.0, 1.0);CHKERRQ(ierr);
  //ierr = SkewArray2D(dm, 45.0);CHKERRQ(ierr);

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
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, dim*conesize, refArray, &refCoords);CHKERRQ(ierr);
  for (i = 0; i < vsize; i++) {
    PetscInt	vertex = vptr[i];
    PetscInt	*points, *foundcells;
    PetscInt	numPoints, j, actualj, cell, k;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
    printf("VERTEX# : %d -> (%.1f , %.1f) ", vertex, coordArray[2*i], coordArray[2*i+1]);
    PetscIntView(2*numPoints, points, 0);
    printf("--------\n");
    ierr = PetscCalloc1(4, &foundcells);CHKERRQ(ierr);
    k = 0;
    for (j = 0; j < numPoints; j++) {
      actualj = 2*j;
      cell = points[actualj];
      if (cell < cEnd) {
        foundcells[k] = cell;
        k++;
      }
    }
    printf("For Vertex %d found %d cells\n", vertex, k);
    ierr = PetscCalloc1(k*dim*dim, &BIGR2Xmat);CHKERRQ(ierr);
    ierr = PetscCalloc1(k*dim*dim, &BIGX2Rmat);CHKERRQ(ierr);
    for (j = 0; j < k; j++) {
      PetscScalar	*R2Xmat, *X2Rmat;

      ierr = PetscCalloc1(dim*dim, &R2Xmat);CHKERRQ(ierr);
      ierr = PetscCalloc1(dim*dim, &X2Rmat);CHKERRQ(ierr);
      printf("cell!: %d\n", cell);
      ierr = ComputeR2X2RMapping(dm, vertex, cell, R2Xmat, X2Rmat);CHKERRQ(ierr);
      ierr = PetscArraycpy(&BIGR2Xmat[j*dim*dim], &R2Xmat, dim*dim);CHKERRQ(ierr);
      ierr = PetscArraycpy(&BIGX2Rmat[j*dim*dim], &X2Rmat, dim*dim);CHKERRQ(ierr);
      ierr = PetscFree(R2Xmat);CHKERRQ(ierr);
      ierr = PetscFree(X2Rmat);CHKERRQ(ierr);
    }
    PetscRealView(k*dim*dim, BIGR2Xmat, 0);
    printf("--------\n");
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
  }

  //VecView(coords, 0);
  //VecView(refCoords, 0);
  //DMView(dm, 0);
  //ISView(globalCellIS, 0);
  ierr = DMPlexComputeGeometryFVM(dm, &cellGeom, &faceGeom);CHKERRQ(ierr);
  //VecView(cellGeom,0);
  //VecView(faceGeom,0);

  ierr = PetscFree(BIGX2Rmat);CHKERRQ(ierr);
  ierr = PetscFree(BIGR2Xmat);CHKERRQ(ierr);
  ierr = VecDestroy(&refCoords);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}
