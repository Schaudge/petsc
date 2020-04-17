#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

static PetscErrorCode PetscSpaceSetFromOptions_Sum(PetscOptionItems *PetscOptionsObject,PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       Ns,Nc,Nv,deg;
  PetscBool      orthogonal = PETSC_TRUE;
  const char     *prefix;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceGetNumVariables(sp,&Nv);CHKERRQ(ierr);
  if (!Nv) PetscFunctionReturn(0);
  ierr = PetscSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp,&Ns);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(sp,&deg,NULL);CHKERRQ(ierr);
  Ns   = (Ns == PETSC_DEFAULT) ? 1 : Ns;

  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSpace sum options");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-petscspace_sum_spaces","The number of subspaces","PetscSpaceSumSetNumSubspaces",Ns,&Ns,NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscspace_sum_orthogonal","Subspaces are orthogonal components of the final space","PetscSpaceSumSetFromOptions",orthogonal,&orthogonal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  if (Ns < 0 || (Nv > 0 && Ns == 0)) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have a sum space of %D spaces\n",Ns);
  if (Ns != sum->numSumSpaces) {
    ierr = PetscSpaceSumSetNumSubspaces(sp,Ns);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetOptionsPrefix((PetscObject)sp,&prefix);CHKERRQ(ierr);
  for (PetscInt i=0; i<Ns; ++i) {
    PetscInt sNv;
    PetscSpace subspace;
    
    ierr = PetscSpaceSumGetSubspace(sp,i,&subspace);CHKERRQ(ierr);
    if (!subspace) {
      char subspacePrefix[256];

      ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)sp),&subspace);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)subspace,prefix);CHKERRQ(ierr);
      ierr = PetscSNPrintf(subspacePrefix,256,"subspace%d_",i);CHKERRQ(ierr);
      ierr = PetscObjectAppendOptionsPrefix((PetscObject)subspace,subspacePrefix);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
    }
    ierr = PetscSpaceSetFromOptions(subspace);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(subspace,&sNv);CHKERRQ(ierr);
    if (!sNv) SETERRQ1(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has not been set properly, number of variables is 0.\n",i);
    ierr = PetscSpaceSumSetSubspace(sp,i,subspace);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&subspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumView_Ascii(PetscSpace sp,PetscViewer v)
{
  PetscSpace_Sum *sum       = (PetscSpace_Sum*)sp->data;
  PetscBool      orthogonal = sum->orthogonal;
  PetscInt       Ns         = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (orthogonal) {
    ierr = PetscViewerASCIIPrintf(v,"Sum space of %D orthogonal subspaces\n",Ns);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(v,"Sum space of %D subspaces\n",Ns);CHKERRQ(ierr);
  }
  for (PetscInt i=0; i<Ns; ++i) {
    ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
    ierr = PetscSpaceView(sum->sumspaces[i],v);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Sum(PetscSpace sp,PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscSpaceSumView_Ascii(sp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscBool      orthogonal = PETSC_TRUE;
  PetscInt       Nv,Ns,Nc,sum_Nc = 0,deg = PETSC_MIN_INT,maxDeg = PETSC_MIN_INT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) PetscFunctionReturn(0);

  ierr = PetscSpaceGetNumVariables(sp,&Nv);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSumGetNumSubspaces(sp,&Ns);CHKERRQ(ierr);
  if (Ns == PETSC_DEFAULT) {
    Ns   = 1;
    ierr = PetscSpaceSumSetNumSubspaces(sp,Ns);CHKERRQ(ierr);
  }
  if (!Ns && Nv) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have zero subspaces\n");

  for (PetscInt i=0; i<Ns; ++i) {
    PetscInt   sNv,sNc,iDeg,iMaxDeg;
    PetscSpace si;

    ierr = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
    ierr = PetscSpaceSetUp(si);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(si,&sNv);CHKERRQ(ierr);
    if (sNv != Nv) SETERRQ3(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Subspace %D has %D variables, space has %D.\n",i,sNv,Nv);
    ierr = PetscSpaceGetNumComponents(si,&sNc);CHKERRQ(ierr);
    if (i == 0 && sNc == Nc) orthogonal = PETSC_FALSE;
    sum_Nc += sNc;
    ierr    = PetscSpaceSumGetSubspace(sp,i,&si);CHKERRQ(ierr);
    ierr    = PetscSpaceGetDegree(si,&iDeg,&iMaxDeg);CHKERRQ(ierr);
    deg     = PetscMax(deg,iDeg);
    maxDeg  = PetscMax(maxDeg,iMaxDeg);
  }

  if (orthogonal) {
    if (sum_Nc != Nc) {
      SETERRQ2(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,
               "Total number of subspace components (%D) does not match number of target space components (%D).",sum_Nc,Nc);
    }
  } else {
    if (sum_Nc != Ns*Nc) {
      SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspaces must have same number of components as the target space.");
    }
  }

  sp->degree       = deg;
  sp->maxDegree    = maxDeg;
  sum->orthogonal  = orthogonal;
  sum->setupCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Sum(PetscSpace sp)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (PetscInt i=0; i<Ns; ++i) {
    ierr = PetscSpaceDestroy(&sum->sumspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(sum);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Sum(PetscSpace sp,PetscInt *dim)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscInt       d = 0,Ns = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);

  for (PetscInt i=0; i<Ns; ++i) {
    PetscInt id;

    ierr = PetscSpaceGetDimension(sum->sumspaces[i],&id);CHKERRQ(ierr);
    d   += id;
  }

  *dim = d;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Sum(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;
  PetscBool      orthogonal = sum->orthogonal;
  DM             dm = sp->dm;
  PetscInt       Nc = sp->Nc,Nv = sp->Nv;
  PetscInt       Ns,pdimfull,numelB,numelD,numelH;
  PetscReal      *sB = NULL,*sD = NULL,*sH = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sum->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  }
  Ns    = sum->numSumSpaces;
  ierr  = PetscSpaceGetDimension(sp,&pdimfull);CHKERRQ(ierr);
  if (B || D || H) {
    numelB = npoints*pdimfull*Nc;
    ierr   = DMGetWorkArray(dm,numelB,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  if (D || H) {
    numelD = numelB*Nv;
    ierr   = DMGetWorkArray(dm,numelD,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (H) {
    numelH = numelD*Nv;
    ierr   = DMGetWorkArray(dm,numelH,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (B) {
    for (PetscInt i=0; i<numelB; ++i) B[i] = 0.;
  }
  if (D) {
    for (PetscInt i=0; i<numelD; ++i) D[i] = 0.;
  }
  if (H) {
    for (PetscInt i=0; i<numelH; ++i) H[i] = 0.;
  }

  for (PetscInt s=0,offset=0,ncoffset=0; s<Ns; ++s) {
    PetscInt sNv,spdim,sNc;

    ierr = PetscSpaceGetNumVariables(sum->sumspaces[s],&sNv);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(sum->sumspaces[s],&sNc);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(sum->sumspaces[s],&spdim);CHKERRQ(ierr);
    if (offset + spdim > pdimfull) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_OUTOFRANGE,"Subspace dimensions exceed target space dimension.\n");
    ierr = PetscSpaceEvaluate(sum->sumspaces[s],npoints,points,sB,sD,sH);CHKERRQ(ierr);
    if (B || D || H){
      for (PetscInt p=0; p<npoints; ++p) {
        for (PetscInt i=0; i<spdim; ++i) {
          for (PetscInt c=0; c<sNc; ++c) {
            PetscInt compoffset,BInd,sBInd;

            compoffset = orthogonal ? c+ncoffset : c;
            BInd       = (p*pdimfull + i + offset)*Nc + compoffset;
            sBInd      = (p*spdim + i)*sNc + c;
            if (B) B[BInd] += sB[sBInd];
            if (D || H) {
              for (PetscInt v=0; v<Nv; ++v){
                PetscInt DInd,sDInd;

                DInd  = BInd*Nv + v;
                sDInd = sBInd*Nv + v;
                if (D) D[DInd] +=sD[sDInd];
                if (H) {
                  for (PetscInt v2=0; v2<Nv; ++v2) {
                    PetscInt HInd,sHInd;

                    HInd     = DInd*Nv + v2;
                    sHInd    = sDInd*Nv + v2;
                    H[HInd] += sH[sHInd];
                  }
                }
              }
            }
          }
        }
      }
    }
    offset   += spdim;
    ncoffset += sNc;
  }

  if (H) {
    ierr = DMRestoreWorkArray(dm,numelH,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMRestoreWorkArray(dm,numelD,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (B || D || H) {
    ierr = DMRestoreWorkArray(dm,numelB,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceSumSetOrthogonal - Sets whether or not subspaces form orthogonal components of the space.

  Input Parameters:
  + sp - the function space object
  - orthogonal - are subspaces orthogonal components (true) or direct summands (false)

Level: intermediate
.seealso: PetscSpaceSumGetOrthogonal()
@*/
PetscErrorCode PetscSpaceSumSetOrthogonal(PetscSpace sp, PetscBool orthogonal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumSetOrthogonal_C",(PetscSpace,PetscBool),(sp,orthogonal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 PetscSpaceSumGetOrthogonal - Get the orthogonality of the subspaces

 Input Parameters:
 . sp - the function space object

 Output Parameters:
 . orthogonal - the orthogonality of the subspaces.

Level: intermediate

.seealso: PetscSpaceSumSetOrthogonal()
@*/
PetscErrorCode PetscSpaceSumGetOrthogonal(PetscSpace sp,PetscBool *orthogonal)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetOrthogonal_C",(PetscSpace,PetscBool*),(sp,orthogonal));CHKERRQ(ierr);
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
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
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
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidIntPointer(numSumSpaces,2);
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
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  if (subsp) PetscValidHeaderSpecific(subsp,PETSCSPACE_CLASSID,3);
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
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  PetscValidPointer(subsp,3);
  ierr = PetscTryMethod(sp,"PetscSpaceSumGetSubspace_C",(PetscSpace,PetscInt,PetscSpace*),(sp,s,subsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetOrthogonal_Sum(PetscSpace sp,PetscBool orthogonal)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)sp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change orthogonality after setup called.\n");

  sum->orthogonal = orthogonal;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetOrthogonal_Sum(PetscSpace sp,PetscBool* orthogonal)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)sp->data;

  PetscFunctionBegin;
  *orthogonal = sum->orthogonal;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumSetNumSubspaces_Sum(PetscSpace space,PetscInt numSumSpaces)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change number of subspaces after setup called\n");
  if (numSumSpaces == Ns) PetscFunctionReturn(0);
  if (Ns >= 0) {
    PetscInt s;
    for (s=0; s<Ns; ++s) {
      ierr = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);
    }
    ierr = PetscFree(sum->sumspaces);CHKERRQ(ierr);
  }

  Ns   = sum->numSumSpaces = numSumSpaces;
  ierr = PetscCalloc1(Ns,&sum->sumspaces);CHKERRQ(ierr);
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
  PetscInt       Ns   = sum->numSumSpaces;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sum->setupCalled) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Cannot change subspace after setup called\n");
  if (Ns < 0) SETERRQ(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSpaceSumSetNumSubspaces() first\n");
  if (s < 0 || s >= Ns) SETERRQ1(PetscObjectComm((PetscObject)space),PETSC_ERR_ARG_OUTOFRANGE,"Invalid subspace number %D\n",subspace);

  ierr              = PetscObjectReference((PetscObject)subspace);CHKERRQ(ierr);
  ierr              = PetscSpaceDestroy(&sum->sumspaces[s]);CHKERRQ(ierr);
  sum->sumspaces[s] = subspace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSumGetSubspace_Sum(PetscSpace space,PetscInt s,PetscSpace *subspace)
{
  PetscSpace_Sum *sum = (PetscSpace_Sum*)space->data;
  PetscInt       Ns = sum->numSumSpaces;

  PetscFunctionBegin;
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

  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetNumSubspaces_C",PetscSpaceSumGetNumSubspaces_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetNumSubspaces_C",PetscSpaceSumSetNumSubspaces_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetSubspace_C",PetscSpaceSumGetSubspace_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetSubspace_C",PetscSpaceSumSetSubspace_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumGetOrthogonal_C",PetscSpaceSumGetOrthogonal_Sum);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceSumSetOrthogonal_C",PetscSpaceSumSetOrthogonal_Sum);CHKERRQ(ierr);
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
  ierr     = PetscSpaceInitialize_Sum(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Should we get rid of this function and provide PetscSpaceSumSetSubspaces instead. I do not see an equivalent of this function for other PetscSpace
 * classes.*/
PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces,const PetscSpace subspaces[],PetscBool orthogonal,PetscSpace *sumSpace)
{
  PetscInt i,Nv,Nc = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sumSpace){
    ierr = PetscSpaceDestroy(sumSpace);CHKERRQ(ierr);
  }
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)subspaces[0]),sumSpace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*sumSpace,PETSCSPACESUM);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetNumSubspaces(*sumSpace,numSubspaces);CHKERRQ(ierr);
  ierr = PetscSpaceSumSetOrthogonal(*sumSpace,orthogonal);CHKERRQ(ierr);
  for(i=0; i<numSubspaces; ++i){
    PetscInt sNc;

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
