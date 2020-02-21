#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp, PetscViewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Sum(PetscSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace sp, PetscInt numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace sp, PetscInt *numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace sp, PetscInt s, PetscSpace subsp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace sp, PetscInt s, PetscSpace *subsp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetNumSubspaces_Sum(PetscSpace space, PetscInt numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetNumSubspaces_Sum(PetscSpace space, PetscInt *numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetSubspace_Sum(PetscSpace space, PetscInt s, PetscSpace subspace)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space, PetscInt s, PetscSpace *subspace)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Sum(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACESUM = "sum" - A PetscSpace object that encapsulates a sum of subspaces.
                  That sum can either be direct, so that the range is the same as both A and B,
                  or orthogonal, so that the range is the concatenation of their ranges,
                  but they both need to be defined on the same number of variables.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&sum);CHKERRQ(ierr);
  sp->data = sum;

  ierr = PetscSpaceInitialize_Sum(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces, const PetscSpace subspaces[], PetscBool orthogonal, PetscSpace *sumSpace)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
