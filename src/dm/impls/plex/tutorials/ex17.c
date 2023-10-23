static const char help[] = "Test of PETSc CAD Shape Optimization & Mesh Modification Technology";

#include <petscdmplex.h>
#include <petsc/private/hashmapi.h>

static PetscErrorCode surfArea(DM dm)
{
  DMLabel   bodyLabel, faceLabel;
  double    surfaceArea = 0., volume = 0.;
  PetscReal vol, centroid[3], normal[3];
  PetscInt  dim, cStart, cEnd, fStart, fEnd;
  PetscInt  bodyID, faceID;
  MPI_Comm  comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscPrintf(comm, "    dim = %" PetscInt_FMT "\n", dim));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));

  if (dim == 2) {
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (PetscInt ii = cStart; ii < cEnd; ++ii) {
		  PetscCall(DMLabelGetValue(faceLabel, ii, &faceID));
		  if (faceID >= 0) {
	      PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
	      surfaceArea += vol;
		  }
	  }
  }

  if (dim == 3) {
	  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
	  for (PetscInt ii = fStart; ii < fEnd; ++ii) {
		  PetscCall(DMLabelGetValue(faceLabel, ii, &faceID));
		  if (faceID >= 0) {
		    PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
		    surfaceArea += vol;
		  }
	  }

	  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
	  for (PetscInt ii = cStart; ii < cEnd; ++ii) {
		  PetscCall(DMLabelGetValue(bodyLabel, ii, &bodyID));
		  if (bodyID >= 0) {
		    PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
		    volume += vol;
		  }
	  }
  }

  if (dim == 2) {
	  PetscCall(PetscPrintf(comm, "    Surface Area = %.6e \n\n", (double)surfaceArea));
  } else if (dim == 3) {
	  PetscCall(PetscPrintf(comm, "    Volume = %.6e \n", (double)volume));
	  PetscCall(PetscPrintf(comm, "    Surface Area = %.6e \n\n", (double)surfaceArea));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  /* EGADS/EGADSlite variables */
  PetscGeom   model, *bodies, *fobjs;
  int         Nb, Nf, id;
  /* PETSc variables */
  DM          dmNozzle = NULL;
  MPI_Comm    comm;
  PetscInt    maxLoopNum = 200;
  PetscScalar equivR = 0.0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dmNozzle));
  PetscCall(DMSetType(dmNozzle, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(dmNozzle, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dmNozzle));

  // Refines Surface Mesh per option -dm_refine
  PetscCall(DMSetFromOptions(dmNozzle));
  PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view"));
  PetscCall(surfArea(dmNozzle));

  for (PetscInt saloop = 0; saloop < maxLoopNum + 1; ++saloop) {
    PetscContainer  modelObj, cpHashTableObj, wHashTableObj, cpCoordDataObj, wDataObj, cpCoordDataLengthObj, wDataLengthObj;
    PetscContainer  gradSACPObj, gradSAWObj;
    PetscContainer  cpEquivObj, maxNumRelateObj;
    PetscScalar    *cpCoordData, *wData, *gradSACP, *gradSAW, *gradVolCP, *gradVolW;
    PetscInt        cpArraySize, wArraySize;
    PetscInt       *cpCoordDataLengthPtr, *wDataLengthPtr, *maxNumRelatePtr;
    PetscHMapI      cpHashTable, wHashTable, cpSurfGradHT;
    Mat             cpEquiv, cpSurfGrad;
    char            stpName[50];

    if (saloop == 0) {
      PetscCall(PetscStrcpy(stpName, "newGeometry_clean_1.stp"));
      PetscCall(DMSetFromOptions(dmNozzle));
    }
    if (saloop == 1) PetscCall(PetscStrcpy(stpName, "newGeometry_clean_2.stp"));

    // Expose Geometry Definition Data and Calculate Surface Gradients
    PetscCall(DMPlexGeomDataAndGrads(dmNozzle, PETSC_FALSE));

    PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "EGADS Model", (PetscObject *)&modelObj));
    if (!modelObj) PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "EGADSLite Model", (PetscObject *)&modelObj));

    // Get attached EGADS model (pointer)
    PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));
    
    // Get Geometric Data from CAD Model attached to DMPlex
    PetscCall(DMPlexGetGeomCntrlPntAndWeightData(dmNozzle, &cpHashTable, &cpCoordDataLength, &cpCoordData, &maxNumEquiv, &cpEquiv, &wHashTable, &wDataLength, &wData));
    PetscCall(DMPlexGetGeomGradData(dmNozzle, &cpSurfGradHT, &cpSurfGrad, &cpArraySize, &gradSACP, &gradVolCP, &wArraySize, &gradSAW, &gradVolW));

    // Get the number of bodies and body objects in the model
    PetscCall(DMPlexGetGeomModelBodies(dmNozzle, &bodies, &Nb));

    // Get all FACES of the Body
    PetscGeom body = bodies[0];
    PetscCall(DMPlexGetGeomModelBodyFaces(dmNozzle, body, &fobjs, &Nf));

    // Print out Surface Area and Volume using EGADS' Function
    PetscScalar volume, surfaceArea;
    PetscInt    COGsize, IMCOGsize;
    PetscScalar *centerOfGravity, *inertiaMatrixCOG;
    PetscCall(DMPlexGetGeomBodyMassProperties(dmNozzle, body, &volume, &surfaceArea, &centerOfGravity, &COGsize, &inertiaMatrixCOG, &IMCOGsize));

    // Calculate Radius of Sphere for Equivalent Volume
    if (saloop == 0) equivR = PetscPowReal(volume * (3./4.) / PETSC_PI, 1./3.);

    // Get Size of Control Point Equivalancy Matrix
    PetscInt numRows, numCols;
    PetscCall(MatGetSize(cpEquiv, &numRows, &numCols));

    // Get max number of relations
    PetscInt maxRelateNew = 0;
    for (PetscInt ii = 0; ii < numRows; ++ii) {
      PetscInt maxRelateNewTemp = 0;
      for (PetscInt jj = 0; jj < numCols; ++jj) {
        PetscScalar matValue;
        PetscCall(MatGetValue(cpEquiv, ii, jj, &matValue));

        if (matValue > 0.0) maxRelateNewTemp += 1;
      }
      if (maxRelateNewTemp > maxRelateNew) maxRelateNew = maxRelateNewTemp;
    }

    // Update Control Point and Weight definitions for each surface
    for (PetscInt jj = 0; jj < Nf; ++jj) {
      PetscGeom face = fobjs[jj];
      PetscInt  numCntrlPnts = 0;
      PetscCall(DMPlexGetGeomID(dmNozzle, body, face, &id));
      PetscCall(DMPlexGetGeomFaceNumOfControlPoints(dmNozzle, face, &numCntrlPnts));

      // Update Control Points
      PetscHashIter CPiter, Witer;
      PetscBool     CPfound, Wfound;
      PetscInt      faceCPStartRow, faceWStartRow;

      PetscCall(PetscHMapIFind(cpHashTable, id, &CPiter, &CPfound));
      PetscCheck(CPfound, comm, PETSC_ERR_SUP, "FACE ID not found in Control Point Hash Table");
      PetscCall(PetscHMapIGet(cpHashTable, id, &faceCPStartRow));

      PetscCall(PetscHMapIFind(wHashTable, id, &Witer, &Wfound));
      PetscCheck(Wfound, comm, PETSC_ERR_SUP, "FACE ID not found in Control Point Weights Hash Table");
      PetscCall(PetscHMapIGet(wHashTable, id, &faceWStartRow));

      for (PetscInt ii = 0; ii < numCntrlPnts; ++ii) {
        PetscReal xold, yold, zold, rold, phi, theta;

        // Update Control Points - Original Code
        xold  = cpCoordData[faceCPStartRow + (3 * ii) + 0];
        yold  = cpCoordData[faceCPStartRow + (3 * ii) + 1];
        zold  = cpCoordData[faceCPStartRow + (3 * ii) + 2];
        rold  = PetscSqrtReal(xold * xold + yold * yold + zold * zold);
        phi   = PetscAtan2Real(yold, xold);
        theta = PetscAtan2Real(PetscSqrtReal(xold * xold + yold * yold), zold);

        // Account for Different Weights for Control Points on Degenerate Edges (multiple control points have same location and different weights
        // only use the largest weight
        PetscScalar maxWeight = 0.0;
        PetscInt    wCntr = 0;
        for (PetscInt kk = faceWStartRow; kk < faceWStartRow + numCntrlPnts; ++kk) {
          PetscScalar matValue;
          PetscCall(MatGetValue(cpEquiv, faceWStartRow + ii, kk , &matValue));

          PetscScalar weight = 0.0;
          if (matValue > 0.0) {
            weight = wData[kk];

            if (weight > maxWeight) {maxWeight = weight;}
            wCntr += 1;
          }
        }

        //Reduce to Constant R = 0.0254
        PetscScalar deltaR;
        PetscScalar localTargetR;
        localTargetR = equivR / maxWeight;
        deltaR       = rold - localTargetR;

        cpCoordData[faceCPStartRow + (3 * ii) + 0] += -0.05 * deltaR * PetscCosReal(phi) * PetscSinReal(theta);
        cpCoordData[faceCPStartRow + (3 * ii) + 1] += -0.05 * deltaR * PetscSinReal(phi) * PetscSinReal(theta);
        cpCoordData[faceCPStartRow + (3 * ii) + 2] += -0.05 * deltaR * PetscCosReal(theta);
      }
    }
    PetscCall(DMPlexFreeGeomObject(dmNozzle, fobjs));
    PetscBool writeFile = PETSC_FALSE;

    // Modify Control Points of Geometry
    PetscCall(DMPlexModifyGeomModel(dmNozzle, comm, cpCoordData, wData, PETSC_FALSE, writeFile, stpName));
    writeFile = PETSC_FALSE;

    // Get attached EGADS model (pointer)
    PetscGeom newmodel;
    PetscCall(PetscContainerGetPointer(modelObj, (void **)&newmodel));

    // Get the number of bodies and body objects in the model
    PetscCall(DMPlexGetGeomModelBodies(dmNozzle, &bodies, &Nb));

    // Get all FACES of the Body
    PetscGeom newbody = bodies[0];
    PetscCall(DMPlexGetGeomModelBodyFaces(dmNozzle, newbody, &fobjs, &Nf));

    // Save Step File of Updated Geometry at designated iterations
    if (saloop == 1) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_1.stp");
    }
    if (saloop == 2) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_2.stp");
    }
    if (saloop == 5) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_5.stp");
    }
    if (saloop == 20) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_20.stp");
    }
    if (saloop == 50) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_50.stp");
    }
    if (saloop == 100) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_100.stp");
    }
    if (saloop == 150) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_150.stp");
    }
    if (saloop == 200) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_200.stp");
    }
    if (saloop == 300) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_300.stp");
    }
    if (saloop == 400) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_400.stp");
    }
    if (saloop == 500) {
      writeFile = PETSC_TRUE;
      PetscStrcpy(stpName, "newGeom_clean_500.stp");
    }

    // Modify Geometry and Inflate Mesh to New Geoemetry
    PetscCall(DMPlexModifyGeomModel(dmNozzle, comm, cpCoordData, wData, PETSC_FALSE, writeFile, stpName));
    PetscCall(DMPlexInflateToGeomModel(dmNozzle, PETSC_TRUE));

    // Periodically Refine and Write Mesh to hdf5 file
    if (saloop == 0)   {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view7"));}
    if (saloop == 1)   {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view8"));}
    if (saloop == 5)   {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view10"));}
    if (saloop == 20)  {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view11"));}
    if (saloop == 50)  {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view12"));}
    if (saloop == 100) {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view13"));}
    if (saloop == 150) {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view14"));}
    if (saloop == 200) {
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view15"));

      PetscCall(DMSetFromOptions(dmNozzle));
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view22"));

      PetscCall(DMSetFromOptions(dmNozzle));
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view23"));
    }
    if (saloop == 300) {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view17"));}
    if (saloop == 400) {PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view19"));}
    if (saloop == 500) {
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view21"));

      PetscCall(DMSetFromOptions(dmNozzle));
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view22"));

      PetscCall(DMSetFromOptions(dmNozzle));
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view23"));
    }
    PetscCall(DMPlexFreeGeomObject(dmNozzle, fobjs));
  }

  /* Close EGADSlite file */
  PetscCall(PetscFinalize());
}

/*TEST

  test:
    suffix: minSA
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/abstract_minSA.stp \
          -dm_refine 1 -dm_plex_geom_print_model 1 -dm_plex_geom_shape_opt 1 \
          -dm_view hdf5:mesh_minSA_abstract.h5 \
          -dm_view1 hdf5:mesh_minSA_vol_abstract.h5 \
          -dm_view2 hdf5:mesh_minSA_vol_abstract_inflated.h5 \
          -dm_view3 hdf5:mesh_minSA_vol_abstract_Refine.h5 \
          -dm_view4 hdf5:mesh_minSA_vol_abstract_Refine2.h5 \
          -dm_view5 hdf5:mesh_minSA_vol_abstract_Refine3.h5 \
          -dm_view6 hdf5:mesh_minSA_vol_abstract_Refine4.h5 \
          -dm_view7 hdf5:mesh_minSA_itr1.h5 \
          -dm_view8 hdf5:mesh_minSA_itr2.h5 \
          -dm_view10 hdf5:mesh_minSA_itr5.h5 \
          -dm_view11 hdf5:mesh_minSA_itr20.h5 \
          -dm_view12 hdf5:mesh_minSA_itr50.h5 \
          -dm_view13 hdf5:mesh_minSA_itr100.h5 \
          -dm_view14 hdf5:mesh_minSA_itr150.h5 \
          -dm_view15 hdf5:mesh_minSA_itr200.h5 \
          -dm_view22 hdf5:mesh_minSA_itr200r1.h5 \
          -dm_view23 hdf5:mesh_minSA_itr200r2.h5

TEST*/
