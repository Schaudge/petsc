static const char help[] = "Test of PETSC/CAD Shape Modification Technology";

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
  /* EGADSLite variables */
  PetscGeom      model, *bodies, *fobjs;
  int            Nb, Nf, id;
  /* PETSc variables */
  DM             dm;
  MPI_Comm       comm;
  PetscContainer modelObj, cpHashTableObj, wHashTableObj, cpCoordDataObj, wDataObj, cpCoordDataLengthObj, wDataLengthObj;
  PetscScalar   *cpCoordData, *wData;
  PetscInt       cpCoordDataLength = 0, wDataLength = 0;
  PetscInt      *cpCoordDataLengthPtr, *wDataLengthPtr;
  PetscHMapI     cpHashTable, wHashTable;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));

  // Refines Surface Mesh per option -dm_refine
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view1"));		// Use this one for surface only meshes
  PetscCall(surfArea(dm));

  // Expose Geometry Definition Data and Calculate Surface Gradients
  PetscCall(DMPlexGeomDataAndGrads(dm, PETSC_FALSE));

  // Get Attached EGADS model
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));

  // Get attached EGADS model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **) &model));

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Hash Table", (PetscObject *)&cpHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinates", (PetscObject *)&cpCoordDataObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject *)&cpCoordDataLengthObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject *)&wHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data", (PetscObject *)&wDataObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&wDataLengthObj));

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  PetscCall(PetscContainerGetPointer(cpHashTableObj, (void **)&cpHashTable));
  PetscCall(PetscContainerGetPointer(cpCoordDataObj, (void **)&cpCoordData));
  PetscCall(PetscContainerGetPointer(cpCoordDataLengthObj, (void **)&cpCoordDataLengthPtr));
  PetscCall(PetscContainerGetPointer(wHashTableObj, (void **)&wHashTable));
  PetscCall(PetscContainerGetPointer(wDataObj, (void **)&wData));
  PetscCall(PetscContainerGetPointer(wDataLengthObj, (void **)&wDataLengthPtr));

  cpCoordDataLength = *cpCoordDataLengthPtr;
  wDataLength       = *wDataLengthPtr;

  // Get the number of bodies and body objects in the model
  PetscCall(DMPlexGetGeomModelBodies(dm, &bodies, &Nb));

  // Get all FACES of the Body
  PetscGeom body = bodies[0];
  PetscCall(DMPlexGetGeomModelBodyFaces(dm, body, &fobjs, &Nf));

  // Update Control Point and Weight definitions for each surface
  for (PetscInt jj = 0; jj < Nf; ++jj) {
    PetscGeom face         = fobjs[jj];
    PetscInt  numCntrlPnts = 0;

    PetscCall(DMPlexGetGeomID(dm, body, face, &id));
    PetscCall(DMPlexGetGeomFaceNumOfControlPoints(dm, face, &numCntrlPnts));

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
	    if (ii == 4) {
		    // Update Control Points - Change the location of the center control point of the faces
        // Note :: Modification geometry requires knowledge of how the geometry is defined.
		    cpCoordData[faceCPStartRow + (3 * ii) + 0] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 0];
		    cpCoordData[faceCPStartRow + (3 * ii) + 1] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 1];
		    cpCoordData[faceCPStartRow + (3 * ii) + 2] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 2];
	    } else {
		    // Do Nothing
        // Note: Could use section the change location of other face control points.
	    }
    }
  }
  PetscCall(DMPlexFreeGeomObject(dm, fobjs));

  // Modify Control Points of Geometry
  PetscCall(DMPlexModifyGeomModel(dm, comm, cpCoordData, wData, PETSC_FALSE, PETSC_TRUE, "newModel_wFunction_clean_20221112.stp"));

  // Inflate Mesh to Geometry
  PetscCall(DMPlexInflateToGeomModel(dm, PETSC_FALSE));
  PetscCall(surfArea(dm));

  // Output .hdf5 file
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view2"));

  // Perform 1st Refinement on the Mesh attached to the new geometry
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view3"));
  PetscCall(surfArea(dm));

  // Perform 2nd Refinement on the Mesh attached to the new geometry
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view4"));
  PetscCall(surfArea(dm));

  // Perform 3 Refinement on the Mesh attached to the new geometry
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view5"));
  PetscCall(surfArea(dm));

   // Perform 4 Refinement on the Mesh attached to the new geometry
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view6"));
  PetscCall(surfArea(dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: sphere_shapeMod
    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sphere_example.stp \
          -dm_refine 1 -dm_plex_geom_print_model 1 -dm_plex_geom_shape_opt 1 \
          -dm_view1 hdf5:mesh_shapeMod_sphere.h5 \
          -dm_view2 hdf5:mesh_shapeMod_inflated.h5 \
          -dm_view3 hdf5:mesh_shapeMod_inflated_Refine1.h5 \
          -dm_view4 hdf5:mesh_shapeMod_inflated_Refine2.h5 \
          -dm_view5 hdf5:mesh_shapeMod_inflated_Refine3.h5 \
          -dm_view6 hdf5:mesh_shapeMod_inflated_Refine4.h5

TEST*/
