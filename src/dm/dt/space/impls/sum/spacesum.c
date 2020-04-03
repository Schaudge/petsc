#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       Ns,Nc,Nv,deg,i;
  PetscBool      orthogonal = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp,&Nv);CHKERRQ(ierr);
  if (!Nv) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp,&Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp,&deg,NULL);CHKERRQ(ierr);
  if (Ns > 1) {
    PetscSpace s0;
    PetscInt   sNc;
    ierr = PetscSpaceSumGetSubspace(sp,0,&s0);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(s0,&sNc);CHKERRQ(ierr);
    if (sNc != Nc) orthogonal = PETSC_TRUE;
  }
  Ns   = (Ns == PETSC_DEFAULT) ? 1 : Ns;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace sum options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscspace_sum_spaces","The number of subspaces","PetscSpaceSumSetNumSubspaces",Ns,&Ns,NULL,
                                0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_sum_orthogonal","Subspaces are orthogonal components of the final space",
                          "PetscSpaceSumSetFromOptions",orthogonal,&orthogonal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (Ns < 0 || (Nv > 0 && Ns == 0)) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a sum space of %D spaces\n",Ns);
  if (Ns != sum->numSumSpaces) {ierr = PetscSpaceSumSetNumSubspaces(sp,Ns);CHKERRQ(ierr);}
  for (i=0; i<Ns; ++i) {
    PetscSpace subspace;
    ierr = PetscSpaceSumGetSubspace(sp,i,&subspace);CHKERRQ(ierr);
    if (!subspace) {
      SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Must provide subspaces to be summed.\n");
    } else {
      ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
    }
    ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
    ierr = PetscSpaceSumSetSubspace(sp,i,subspace);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp,PetscViewer v)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *) sp->data;
  PetscBool          orthogonal = sum->orthogonal;
  PetscInt           Ns = sum->numSumSpaces,i;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (orthogonal) {
    ierr = PetscViewerASCIIPrintf(v, "Sum space of %D orthogonal subspaces\n", Ns);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(v, "Sum space of %D subspaces\n", Ns);CHKERRQ(ierr);
  }
  for (i = 0; i < Ns; i++) {
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscSpaceView(sum->sumspaces[i], v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp,PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpaceSumView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum    = (PetscSpace_Sum *) sp->data;
  PetscInt           Nv, Ns, Nc, sNc, sum_Nc, i;
  PetscBool          orthogonal = PETSC_TRUE;
  PetscInt           deg, maxDeg;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumVariables(sp, &Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp, &Ns);CHKERRQ(ierr);
  if (Ns == PETSC_DEFAULT) {
    Ns = 1;
    ierr = PetscSpaceSumSetNumSubspaces(sp, Ns);CHKERRQ(ierr);
  }
  if (!Ns) {
    if (Nv) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_OUTOFRANGE, "Cannot have zero subspaces\n");
  } else {
    /* We need to ensure that the subspaces have been created/setup before we call GetNumComponents, but we also need to be
     * able to determine how to create the subspaces since the number of subpace components will help us to determine if orthogonal. */
    PetscSpace s0, si;
    ierr = PetscSpaceSumGetSubspace(sp, 0, &s0);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(s0, &sNc);CHKERRQ(ierr); /* ex8 Error Here.*/
    if (sNc == Nc) orthogonal = PETSC_FALSE;
    if (orthogonal) {
      sum_Nc = sNc;
      for (i=1; i<Ns; ++i){
        ierr = PetscSpaceSumGetSubspace(sp, i, &si);CHKERRQ(ierr);
        ierr = PetscSpaceGetNumComponents(si, &sNc);CHKERRQ(ierr);
        sum_Nc += sNc;
      }
      if (sum_Nc != Nc) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE, "Total number of subspace components (%D) does not match number of target space components (%D).",sum_Nc,Nc);

    } else {
      for (i = 1 ; i < Ns; i++) {
        ierr = PetscSpaceSumGetSubspace(sp, i, &si);CHKERRQ(ierr);
        ierr = PetscSpaceGetNumComponents(si, &sNc);CHKERRQ(ierr);
        if (sNc != Nc) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspaces must have same number of components as the target space.");
      }
    }
  }
  deg = PETSC_MIN_INT;
  maxDeg = PETSC_MIN_INT;
  for (i = 0; i < Ns; i++) {
    PetscSpace si;
    PetscInt   iDeg, iMaxDeg;

    ierr   = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
    ierr   = PetscSpaceGetDegree(si,&iDeg,&iMaxDeg);CHKERRQ(ierr);
    /* The summed space could potentially contain a bigger space than any of its components. If this is the case the user will need to set override
     * the degree using PetscSpaceSetDegree. */
    deg    = PetscMax(deg,iDeg);
    maxDeg = PetscMax(maxDeg,iMaxDeg);
  }
  sp->degree    = deg;
  sp->maxDegree = maxDeg;
  sum->orthogonal = orthogonal;
  sum->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       Ns,i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Ns = sum->numSumSpaces;
  for (i=0; i<Ns; ++i) {ierr = PetscSpaceDestroy(&sum->sumspaces[i]);CHKERRQ(ierr);}
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceSumSetSubspace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceSumGetSubspace_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceSumSetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpaceSumGetNumSubspaces_C", NULL);CHKERRQ(ierr);
  ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  ierr = PetscFree(sum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp,PetscInt *dim)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       i,Ns,d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  Ns = sum->numSumSpaces;
  d  = 0;
  /* This works for both orthogonal and non-orthogonal cases so long as we assume that non-orthogonal
   * summands have no overlapping basis components.... Good assumption? Probably need a check for this somewhere just to be safe.*/
  for (i = 0; i < Ns; i++) {
    PetscInt id;

    ierr = PetscSpaceGetDimension(sum->sumspaces[i], &id);CHKERRQ(ierr);
    d += id;
  }
  *dim = d;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  DM             dm   = sp->dm;
  PetscInt       Nc   = sp->Nc;
  PetscInt       Nv   = sp->Nv;
  PetscInt       Ns;
  PetscReal      *sB = NULL,*sD = NULL,*sH = NULL;
  PetscInt       c,v,v2,pdim,pdimfull,d,i,p,s,offset,ncoffset,compoffset;
  PetscBool      orthogonal = sum->orthogonal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sum->setupCalled) {ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);}
  Ns    = sum->numSumSpaces;
  ierr  = PetscSpaceGetDimension(sp,&pdimfull);CHKERRQ(ierr);
  pdim  = pdimfull / Nc;
  /*When doing the orthogonal sum, there is the possibility that sNc != Nc. Then we will have allocated more memory than we need for sB,
   * sD, sH. I don't think this causes any problems in terms of the array access
   * operations in the evaluation code below but we could be more precise if we allocate the proper size for each subspace. This would also mean
   * though that in the non-orthogonal case we are re-allocating the same size each time which is also less
   * efficient. So maybe 6 to one, half-dozen the other?*/
  if (B || D || H) {ierr = DMGetWorkArray(dm,npoints*Nc*pdimfull,MPIU_REAL,&sB);CHKERRQ(ierr);}
  if (D || H) {ierr = DMGetWorkArray(dm,npoints*Nc*pdimfull*Nv,MPIU_REAL,&sD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm,npoints*Nc*pdimfull*Nv*Nv,MPIU_REAL,&sH);CHKERRQ(ierr);}
  /* We assume here that the caller has already allocated B, D, and H to be the proper size, is this a good assumption? */
  if (B) {
    for (i=0; i<npoints*Nc*pdimfull; ++i) B[i] = 0.;
  }
  if (D) {
    for (i=0; i<npoints*Nc*pdimfull*Nv; ++i) D[i] = 0.;
  }
  if (H) {
    for (i=0; i<npoints*Nc*pdimfull*Nv*Nv; ++i) H[i] = 0.;
  }
  for (s=0,d=0,offset=0,ncoffset=0; s<Ns; ++s) {
    PetscInt sNv,spdim,sNc;

    ierr = PetscSpaceGetNumVariables(sum->sumspaces[s],&sNv);CHKERRQ(ierr);
    if (sNv != Nv) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot create sumspace with different number of variables than its summands. Space requires %D variables, subspace has %D.\n",Nv,sNv);
    ierr = PetscSpaceGetNumComponents(sum->sumspaces[s],&sNc);CHKERRQ(ierr);
    if (!orthogonal && sNc != Nc) SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot create sumspace with different number of components than its summands. Space has %D components, subspace has %D.\n",Nc,sNc);
    ierr = PetscSpaceGetDimension(sum->sumspaces[s],&spdim);CHKERRQ(ierr);
    if (offset+spdim > pdimfull) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE, "Subspace dimensions exceed target space dimension.\n");
    ierr = PetscSpaceEvaluate(sum->sumspaces[s],npoints,points,sB,sD,sH);CHKERRQ(ierr);
    if (B || D || H){
      for (p=0; p<npoints; ++p) {
        for (c=0; c<sNc; ++c) {
          compoffset = orthogonal ? c+ncoffset : c;
          for (i=0; i<spdim; ++i) {
            /* Could possibly save a few flops here by pre-computing common parts of the array indices instead of doing
             * all the multiplications on the fly. Micro-optimization?? */
            if (B) B[(p*Nc + compoffset)*pdimfull + i+offset] += sB[(p*sNc+c)*spdim + i];
            if (D || H) {
              for (v=0; v<Nv; ++v){
                if (D) D[((p*Nc+compoffset)*Nv +v)*pdimfull + i + offset] +=sD[((p*sNc+c)*Nv +v)*spdim + i];
                if (H) {
                  for (v2=0; v2<Nv; ++v2) H[(((p*Nc + compoffset)*Nv+v)*Nv + v2)*pdimfull + i +offset] += sH[(((p*sNc +c)*Nv+v)*Nv+v2)*spdim + i];
                }
              }
            }
          }
        }
      }
    }
    d += sNv;
    offset += spdim;
    ncoffset += sNc;
  }

  if (H)           {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (B || D || H) {ierr = DMRestoreWorkArray(dm, npoints*pdim,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceSumSetOrthogonal(PetscSpace sp, PetscBool orthogonal)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum *) sp->data;

  PetscFunctionBegin;
  if (sum->setupCalled) {
    SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change orthogonality after setup called.\n");
  }
  sum->orthogonal = orthogonal;

  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetNumSubspaces - Set the number of spaces in the sum

  Input Parameters:
  + sp  - the function space object
  - numSumSpaces - the number of spaces

Level: intermediate

.seealso: PetscSpaceSumGetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace sp,PetscInt numSumSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetNumSubspaces_C",(PetscSpace,PetscInt),(sp,numSumSpaces));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumGetNumSubspaces - Get the number of spaces in the sum

  Input Parameter:
  . sp  - the function space object

  Output Parameter:
  . numSumSpaces - the number of spaces

Level: intermediate

.seealso: PetscSpaceSumSetNumSubspaces(), PetscSpaceSetDegree(), PetscSpaceSetNumVariables()
@*/
PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace sp,PetscInt *numSumSpaces)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidIntPointer(numSumSpaces, 2);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetNumSubspaces_C",(PetscSpace,PetscInt*),(sp,numSumSpaces));CHKERRQ(ierr);
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
  PetscInt i,Nv,Nc,sNc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sumSpace){
    ierr = PetscSpaceDestroy(sumSpace);CHKERRQ(ierr);
  }
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]),sumSpace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*sumSpace,PETSCSPACESUM);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetNumSubspaces(*sumSpace,numSubspaces);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetOrthogonal(*sumSpace,orthogonal);CHKERRQ(ierr);
  Nc = 0;
  for(i=0; i<numSubspaces; ++i){
    ierr = PetscSpaceSumSetSubspace(*sumSpace,i,subspaces[i]);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(subspaces[i],&sNc);CHKERRQ(ierr);
    if (orthogonal) 
    {
      Nc += sNc;
    } else {
      Nc = sNc;
    }
  }
  ierr = PetscSpaceGetNumVariables(subspaces[0],&Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*sumSpace,Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*sumSpace,Nv);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(*sumSpace);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}
