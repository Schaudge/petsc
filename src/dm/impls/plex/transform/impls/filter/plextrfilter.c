#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_Filter(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Filter transformation %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetUp_Filter(DMPlexTransform tr)
{
  DM       dm;
  DMLabel  active;
  PetscInt Nc;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  if (active) {
    IS              filterIS;
    const PetscInt *filterCells;
    PetscInt        c;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Filter Type", &tr->trType));
    PetscCall(DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &filterIS));
    PetscCall(DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc));
    if (filterIS) PetscCall(ISGetIndices(filterIS, &filterCells));
    for (c = 0; c < Nc; ++c) {
      const PetscInt cell    = filterCells[c];
      PetscInt      *closure = NULL;
      DMPolytopeType ct;
      PetscInt       Ncl, cl;

      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
      for (cl = 0; cl < Ncl * 2; cl += 2) {
        PetscCall(DMPlexGetCellType(dm, closure[cl], &ct));
        PetscCall(DMLabelSetValue(tr->trType, closure[cl], ct));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    }
    if (filterIS) {
      PetscCall(ISRestoreIndices(filterIS, &filterCells));
      PetscCall(ISDestroy(&filterIS));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_Filter(DMPlexTransform tr)
{
  DMPlexTransform_Filter *f = (DMPlexTransform_Filter *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_Filter(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscFunctionBeginHot;
  if (tr->trType && p >= 0) {
    PetscInt val;

    PetscCall(DMLabelGetValue(tr->trType, p, &val));
    if (val >= 0) {
      if (rt) *rt = val;
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  if (rt) *rt = -1;
  *Nt     = 0;
  *target = NULL;
  *size   = NULL;
  *cone   = NULL;
  *ornt   = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetDimensions_Filter_Private(DMPlexTransform tr, DM dm, DM tdm)
{
  DMLabel         subpMap;
  IS              valueIS, pointIS;
  const PetscInt *values, *points;
  PetscInt        Nv, Np;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformSetDimensions_Internal(tr, dm, tdm));
  // Create subpoint map
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Subpoint Map", &subpMap));
  PetscCall(DMLabelGetValueIS(tr->trType, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  for (PetscInt v = 0; v < Nv; ++v) {
    PetscCall(DMLabelGetStratumIS(tr->trType, values[v], &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &Np));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt p = 0; p < Np; ++p) {
      PetscCall(DMLabelSetValue(subpMap, points[p], DMPolytopeTypeGetDim((DMPolytopeType)values[v])));
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscCall(DMPlexSetSubpointMap(tdm, subpMap));
  PetscCall(DMLabelDestroy(&subpMap));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_Filter(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_Filter;
  tr->ops->setup                 = DMPlexTransformSetUp_Filter;
  tr->ops->destroy               = DMPlexTransformDestroy_Filter;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Filter_Private;
  tr->ops->celltransform         = DMPlexTransformCellTransform_Filter;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientationIdentity;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Filter(DMPlexTransform tr)
{
  DMPlexTransform_Filter *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_Filter(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
