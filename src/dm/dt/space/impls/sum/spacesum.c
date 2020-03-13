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
  PetscInt       i,Ns,Nc,d;
  PetscBool      orthogonal = sum->orthogonal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  Ns = sum->numSumSpaces;
  Nc = sp->Nc;
  d  = 1;
  for (i = 0; i < Ns; i++) {
    PetscInt id;
    ierr = PetscSpaceGetDimension(sum->sumspaces[i], &id);CHKERRQ(ierr);
    if (orthogonal){
      d += id;  
    } else {
      if (id > d) d = id;
    }
  }
  d *= Nc;
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
  PetscReal      *lpoints,*sB = NULL,*sD = NULL,*sH = NULL;
  PetscInt       c,pdim,d,e,der,der2,i,l,si,p,s,step;
  PetscBool      orthogonal = sum->orthogonal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sum->setupCalled) {ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);}
  Ns    = sum->numSumSpaces;
  ierr  = PetscSpaceGetDimension(sp,&pdim);CHKERRQ(ierr);
  pdim /= Nc;
  ierr  = DMGetWorkArray(dm,npoints*Nv,MPIU_REAL,&lpoints);CHKERRQ(ierr);
  if (orthogonal) {
  /* If we do an orthogonal sum then these arrays all need an extra Nc elements
   * right? */
  } else {
    if (B || D || H) {ierr = DMGetWorkArray(dm,npoints*pdim,MPIU_REAL,&sB);CHKERRQ(ierr);}
    if (D || H) {ierr = DMGetWorkArray(dm,npoints*pdim*Nv,MPIU_REAL,&sD);CHKERRQ(ierr);}
    if (H) {ierr = DMGetWorkArray(dm,npoints*pdim*Nv*Nv,MPIU_REAL,&sH);CHKERRQ(ierr);}
    if (B) {
      for (i=0; i<npoints*pdim*Nc*Nc; ++i) B[i] = 0.;
    }
    if (D) {
      for (i=0; i<npoints*pdim*Nc*Nc*Nv; ++i) D[i] = 0.;
    }
    if (H) {
      for (i=0; i<npoints*pdim*Nc*Nc*Nv*Nv; ++i) H[i] = 0.;
    }
  }
  for (s=0,d=0,step=1; s<Ns; ++s){
    PetscInt sNv,spdim;
    PetscInt skip,j,k;

    ierr = PetscSpaceGetNumVariables(sum->sumspaces[s],&sNv);CHKERRQ(ierr);
    ierr = PetscSpaceGetDimension(sum->sumspaces[s],&spdim);CHKERRQ(ierr);
    if ((pdim % step) || (pdim % spdim)) SETERRQ6 (PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad sum loop: Nv %D, Ns %D, pdim %D, s %D, step %D, spdim %D",Nv,Ns,pdim,s,step,spdim);
    skip = pdim/(step*spdim);
    for (p=0; p<npoints; ++p) {
      for (i=0; i<sNv; ++i) {
        lpoints[p*sNv+i] = points[p*Nv+d+i];
      }
    }
    ierr = PetscSpaceEvaluate(sum->sumspaces[s],npoints,lpoints,sB,sD,sH);CHKERRQ(ierr);
    if (B) {
      for (p=0; p<npoints; ++p) {
        for (k=0; k<skip; ++k) {
          for (si=0; si<spdim; ++si) {
            for (j=0; j<step; ++j) {
              if (orthogonal){
                /* Do orthogonal sum. Probably need to setup the arrays differently as well */
              } else {
                i = (k*spdim+si)*step+j;
                B[(pdim*p+i)*Nc*Nc] += sB[spdim*p+si];
              } 
            }
          }
        }
      }
    }
    if (D) {
      for (p=0; p<npoints; ++p) {
        for (k=0; k<skip; ++k) {
          for (si=0; si<spdim; ++si) {
            for (j=0; j<step ++j){
              i = (k*spdim+si)*step+j;
              for (der=0; der<Nv; ++der){
                if (der >= d && der < d + sNv) {
                  if (orthogonal) {
                    /* Do orthogonal sum. */
                  } else {
                    D[(pdim*p+i)*Nc*Nc*Nv+der] += sD[(spdim*p+si)*sNv+der-d];
                  }
                } else {
                  if (orthogonal) {
                    /* Do orthogonal sum. */
                  } else {
                    D[(pdim*p+i)*Nc*Nc*Nv+der] += sB[spdim*p+si];
                  }
                }
              }
            }
          }
        }
      }
    }
    if (H) {
      for (p=0; p<npoints; ++p) {
        for (k=0; k<skip; ++k) {
          for (si=0; si<spdim; ++si) {
            for (j=0; j<step; ++j) {
              i = (k*spdim+si)*step+j;
              for (der=0; der<Nv; ++der) {
                for (der2=0; der2<Nv; ++der2) {
                  if (der >= d && der < d+sNv && der2 >= d && der2 < d+sNv) {
                    if (orthogonal) {
                     /* Orthogonal calc. */ 
                    } else {
                      H[((pdim*p+i)*Nc*Nc*Nv+der)*Nv+der2] += sH[((spdim*p+si)*sNv+der-d)*sNv+der2-d];
                    }
                  } else if (der >= d && der < d+sNv) {
                    if (orthogonal){
                     /* Orthogonal calc. */
                    } else {
                    H[((pdim*p+i)*Nc*Nc*Nv+der)*Nv+der2] += sD[(spdim*p+si)*sNv+der-d]; 
                    }
                  } else if (der2 >= d && der2 < d + sNv) {
                    if (orthogonal) {
                      /* Orthogonal calc. */
                    } else {
                    H[((pdim*p+i)*Nc*Nc*Nv+der)*Nv+der2] += sD[(spdim*p+si)*sNv+der2-d];
                    }
                  } else {
                    if (orthogonal) {
                      /* Orthogonal calc. */
                    } else {
                      H[((pdim*p+i)*Nc*Nc*Nv+der)*Nv+der2] += sB[spdim*p+si];

                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    d += sNv;
    step *= spdim;
  }

  if (H)           {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv*Nv, MPIU_REAL, &sH);CHKERRQ(ierr);}
  if (D || H)      {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nv,    MPIU_REAL, &sD);CHKERRQ(ierr);}
  if (B || D || H) {ierr = DMRestoreWorkArray(dm, npoints*pdim,       MPIU_REAL, &sB);CHKERRQ(ierr);}
  ierr = DMRestoreWorkArray(dm, npoints*Nv, MPIU_REAL, &lpoints);CHKERRQ(ierr);
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
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
