#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp,PetscViewer v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp,PetscViewer viewer)
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

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp,PetscInt *dim)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace sp,PetscInt numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace sp,PetscInt *numTensSpaces)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetSubspace - Set a space in the sum

  Input Parameters:
+ sp    - the function space object
. s     - The space number
- subsp - the number of spaces

  Level: intermediate

.seealso: PetscSpaceSumGetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace sp,PetscInt s,PetscSpace subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  if (subsp) PetscValidHeaderSpecific(subsp, PETSCSPACE_CLASSID, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetSubspace_C",(PetscSpace,PetscInt,PetscSpace),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumGetSubspace - Get a space in the sum 

  Input Parameters:
+ sp - the function space object
- s  - The space number

  Output Parameter:
. subsp - the PetscSpace

  Level: intermediate

.seealso: PetscSpaceSumSetSubspace(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace sp,PetscInt s,PetscSpace *subsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(subsp, 3);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetNumSubspaces_Sum(PetscSpace space,PetscInt numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *) space->data;
  PetscInt           Ns;
  PetscErrorCode     ierr;
  
  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called\n");
  Ns = sum->numSumSpaces;
  if (numSumSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;
    for (s = 0; s < Ns; s++) {ierr = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);}
    ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  }
  Ns = sum->numSumSpaces = numSumSpaces;
  ierr = PetscCalloc1(Ns, &sum->sumspaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetNumSubspaces_Sum(PetscSpace space,PetscInt *numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;

  PetscFunctionBegin;
  *numSumSpaces = sum->numSumSpaces;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),
                                PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called\n");
  Ns = sum->numSumSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm ((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first\n");
  if (s < 0 || s >= Ns) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D\n",subspace);
  ierr              = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr              = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace *subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns;

  PetscFunctionBegin;
  Ns = sum->numSumSpaces;
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first\n");
  if (s<0 || s>=Ns) SETERRQ1 (PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D\n",subspace);
  *subspace = sum->sumspaces[s];
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Sum(PetscSpace sp)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_Sum;
  sp->ops->setup          = PetscSpaceSetUp_Sum;
  sp->ops->view           = PetscSpaceView_Sum;
  sp->ops->destroy        = PetscSpaceDestroy_Sum;
  sp->ops->getdimension   = PetscSpaceGetDimension_Sum;
  sp->ops->evaluate       = PetscSpaceEvaluate_Sum;
  ierr                    = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",PetscSpaceSumGetNumSubspaces_Sum);CHKERRQ(
    ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",PetscSpaceSumSetNumSubspaces_Sum);CHKERRQ(
    ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",PetscSpaceSumGetSubspace_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",PetscSpaceSumSetSubspace_Sum);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr     = PetscNewLog(sp,&sum);CHKERRQ(ierr);
  sp->data = sum;

  ierr = PetscSpaceInitialize_Sum(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces,const PetscSpace subspaces[],PetscBool orthogonal,PetscSpace *sumSpace)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
