#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@ 
  PetscSpaceKoszulSetDomain - Set the space to use as domain for the koszul map

  Input Parameters:
  + koszulsp    - the koszul space object
  - domainsp    - the space object to use as the domain of the mapping

Level: intermediate

.seealso:
@*/
PetscErrorCode PetscSpaceKoszulSetDomain(PetscSpace koszulsp,PetscSpace domainsp){
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(koszulsp,PETSCSPACE_CLASSID,1);
  if (domainsp) PetscValidHeaderSpecific(domainsp,PETSCSPACE_CLASSID,3);
  ierr = PetscTryMethod(koszulsp,"PetscSpaceKoszulSetDomain_C",(PetscSpace,PetscSpace),(koszulsp,domainsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceKoszulSetDomain_Koszul(PetscSpace koszulsp, PetscSpace domainsp)
{
  PetscSpace_Koszul *kosz = (PetscSpace_Koszul*)koszulsp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (kosz->setupCalled) SETERRQ(PetscObjectComm((PetscObject)koszulsp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change domain space after setup called\n");
  
  ierr = PetscObjectReference((PetscObject)domainsp);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(kosz->domainspace);CHKERRQ(ierr);
  *(kosz->domainspace) = domainsp;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Koszul(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscSpace_Koszul *koszul = (PetscSpace_Koszul*)sp->data;
  DM             dm = sp->dm;
  PetscInt       Nc = sp->Nc,Nv = sp->Nv;
  PetscInt       i,s,offset,ncoffset,pdimfull,numelB,numelD,numelH;
  PetscReal      *sB = NULL,*sD = NULL,*sH = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!koszul->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  }
  ierr   = PetscSpaceGetDimension(sp,&pdimfull);CHKERRQ(ierr);
  numelB = npoints*pdimfull*Nc;
  numelD = numelB*Nv;
  numelH = numelD*Nv;
  if (B || D || H) {
    ierr = DMGetWorkArray(dm,numelB,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMGetWorkArray(dm,numelD,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (H) {
    ierr = DMGetWorkArray(dm,numelH,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (B)
    for (i=0; i<numelB; ++i) B[i] = 0.;
  if (D)
    for (i=0; i<numelD; ++i) D[i] = 0.;
  if (H)
    for (i=0; i<numelH; ++i) H[i] = 0.;

  ierr = PetscSpaceEvaluate(koszul->domainspace,npoints,points,sB,sD,sH);CHKERRQ(ierr);

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

static PetscErrorCode PetscSpaceInitialize_Koszul(PetscSpace sp)
{

  PetscFunctionBegin;
  sp->ops->evaluate       = PetscSpaceEvaluate_Koszul;

  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEKOSZUL = "koszul" - A space that results from application of the koszul operator. 

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Koszul(PetscSpace sp)
{
  PetscSpace_Koszul *koszul;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr     = PetscNewLog(sp,&koszul);CHKERRQ(ierr);
  sp->data = koszul;
  ierr     = PetscSpaceInitialize_Koszul(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateKoszul(PetscSpace *domainsp,PetscSpace *koszulsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (koszulsp) {
    ierr = PetscSpaceDestroy(koszulsp);CHKERRQ(ierr);
  }
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)*domainsp),koszulsp);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*koszulsp,PETSCSPACEKOSZUL);CHKERRQ(ierr);
  /* Inherit number of variables and components from domain space */
  ierr = PetscSpaceKoszulSetDomain(*koszulsp,*domainsp);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*koszulsp,(*domainsp)->Nv);CHKERRQ(ierr);
  /* TODO: number of compontents for a koszul space is potentially different from the
   * domain space since we increase the form degree. This should be known in advance? Maybe we require user to set
   * number of components?*/
  ierr = PetscSpaceSetNumComponents(*koszulsp, (*domainsp)->Nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
