#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@
  PetscSpaceKoszulSetDomain - Set the space to use as domain for the koszul map

  Input Parameters:
  + koszulsp    - the koszul space object
  - domainsp    - the space object to use as the domain of the mapping

Level: intermediate

.seealso:
@*/
PetscErrorCode PetscSpaceKoszulSetDomain(PetscSpace koszulsp,PetscSpace domainsp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(koszulsp,PETSCSPACE_CLASSID,1);
  if (domainsp) PetscValidHeaderSpecific(domainsp,PETSCSPACE_CLASSID,3);
  ierr = PetscTryMethod(koszulsp,"PetscSpaceKoszulSetDomain_C",(PetscSpace,PetscSpace),(koszulsp,domainsp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpaceKoszulSetFormDegree - Set the form degree of the space

  Input Parameters:
  + koszulsp - the koszul space object
  - formDegree - the form degree that is expected after applying the koszul operator

Level: intermediate

.seealso:
@*/
PetscErrorCode PetscSpaceKoszulSetFormDegree(PetscSpace koszulsp,PetscInt formDegree){
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(koszulsp,PETSCSPACE_CLASSID,1);
    ierr = PetscTryMethod(koszulsp,"PetscSpaceKoszulSetFormDegree_C",(PetscSpace,PetscInt),(koszulsp,formDegree));CHKERRQ(ierr);
    PetscFunctionReturn(0);

}

static PetscErrorCode PetscSpaceKoszulSetDomain_Koszul(PetscSpace koszulsp,PetscSpace domainsp)
{
  PetscSpace_Koszul * kosz = (PetscSpace_Koszul*)koszulsp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (kosz->setupCalled) SETERRQ(PetscObjectComm((PetscObject)koszulsp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change domain space after setup called\n");

  ierr                 = PetscObjectReference((PetscObject)domainsp);CHKERRQ(ierr);
  if (kosz->domainspace) {
    ierr = PetscSpaceDestroy(&kosz->domainspace);CHKERRQ(ierr);
  }
  kosz->domainspace = domainsp;
  koszulsp->dim = domainsp->dim;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceKoszulSetFormDegree_Koszul(PetscSpace koszulsp,PetscInt formDegree)
{
  PetscSpace_Koszul * kosz = (PetscSpace_Koszul*)koszulsp->data;

  PetscFunctionBegin;
  if (kosz->setupCalled) SETERRQ(PetscObjectComm((PetscObject)koszulsp),PETSC_ERR_ARG_WRONGSTATE,"Cannot change form degree after setup called\n");
  kosz->formDegree = formDegree;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Koszul(PetscSpace sp,PetscInt npoints,const PetscReal points[],PetscReal B[],PetscReal D[],PetscReal H[])
{
  PetscSpace_Koszul * koszul = (PetscSpace_Koszul*)sp->data;
  DM                dm = sp->dm;
  PetscInt          Nc_k = sp->Nc,Nv = sp->Nv,Nc_d = (koszul->domainspace)->Nc,formDegree = koszul->formDegree;
  PetscInt          i,p,pdimfull_d,pdimfull_k,numelB_d,numelB_k,numelD_d,numelD_k,numelH_d,numelH_k;
  PetscReal         * sB = NULL,*sD = NULL,*sH = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!koszul->setupCalled) {
    ierr = PetscSpaceSetUp(sp);CHKERRQ(ierr);
  }
  /* Set up work arrays*/
  ierr = PetscSpaceGetDimension(sp,&pdimfull_k);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(koszul->domainspace,&pdimfull_d);CHKERRQ(ierr);

  /* In general, we will need different sizes for the domainspace and
   * koszul-applied evaluations */
  numelB_d = npoints * pdimfull_d * Nc_d;
  numelB_k = npoints * pdimfull_k * Nc_k;
  numelD_d = numelB_d * Nv;
  numelD_k = numelB_k*Nv;
  numelH_d = numelD_d * Nv;
  numelH_k = numelD_k*Nv;
  if (B || D || H) {
    ierr = DMGetWorkArray(dm,numelB_d,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMGetWorkArray(dm,numelD_d,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (H) {
    ierr = DMGetWorkArray(dm,numelH_d,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (B) PetscArrayzero(B,numelB_k);
  if (D) PetscArrayzero(D,numelD_k);
  if (H) PetscArrayzero(H,numelH_k);

  /* Evaluate the domain space */
  ierr = PetscSpaceEvaluate(koszul->domainspace,npoints,points,sB,sD,sH);CHKERRQ(ierr);

  /* Now apply koszul operator using AltV interior at each point */
  PetscInt f;
  for (f = 0; f < pdimfull_d; ++f) {
    for (p = 0; p < npoints; ++p) {
      /* apply interior product */
      /* If Nc_k == (N chooose k) then all we need is the call to AltV,
       * otherwise we have to duplicate the entries??? */
      ierr = PetscDTAltVInterior(Nv,formDegree+1,&sB[(f*npoints+p)*Nc_d],&points[p*Nv],&B[(f*npoints+p)*Nc_k]);CHKERRQ(ierr);
    }
  }
  if (H) {
    ierr = DMRestoreWorkArray(dm,numelH_d,MPIU_REAL,&sH);CHKERRQ(ierr);
  }
  if (D || H) {
    ierr = DMRestoreWorkArray(dm,numelD_d,MPIU_REAL,&sD);CHKERRQ(ierr);
  }
  if (B || D || H) {
    ierr = DMRestoreWorkArray(dm,numelB_d,MPIU_REAL,&sB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Koszul(PetscSpace sp)
{
    PetscSpace_Koszul *kosz = (PetscSpace_Koszul*)sp->data;
    PetscErrorCode ierr;

    PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceKoszulSetDomain_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceKoszulSetFormDegree_C",NULL);CHKERRQ(ierr);
    ierr = PetscSpaceDestroy(&kosz->domainspace);CHKERRQ(ierr);
    PetscFree(kosz);

    PetscFunctionReturn(0);

}

static PetscErrorCode PetscSpaceInitialize_Koszul(PetscSpace sp)
{
    PetscErrorCode ierr;
  PetscFunctionBegin;
  sp->ops->evaluate = PetscSpaceEvaluate_Koszul;
  sp->ops->destroy = PetscSpaceDestroy_Koszul;

  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceKoszulSetDomain_C",PetscSpaceKoszulSetDomain_Koszul);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sp,"PetscSpaceKoszulSetFormDegree_C",PetscSpaceKoszulSetFormDegree_Koszul);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEKOSZUL = "koszul" - A space that results from application of the
koszul operator.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Koszul(PetscSpace sp)
{
  PetscSpace_Koszul * koszul;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp,PETSCSPACE_CLASSID,1);
  ierr     = PetscNewLog(sp,&koszul);CHKERRQ(ierr);
  sp->data = koszul;
  ierr     = PetscSpaceInitialize_Koszul(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateKoszul(PetscSpace * domainsp,PetscInt formDegree, PetscInt Nc, PetscSpace * koszulsp)
{
  PetscInt nChooseK;
  PetscSpace_Koszul* kosz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject) *domainsp),koszulsp);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*koszulsp,PETSCSPACEKOSZUL);CHKERRQ(ierr);
  /* Inherit number of variables and components from domain space */
  ierr = PetscSpaceSetNumVariables(*koszulsp,(*domainsp)->Nv);CHKERRQ(ierr);
  /* Gonna set degrees like this for now even though I'm pretty sure we can make
   * 0 guarantees about the approximation properties of the koszul space */
  ierr = PetscSpaceSetDegree(*koszulsp,(*domainsp)->degree,(*domainsp)->maxDegree+1);CHKERRQ(ierr);

  ierr = PetscSpaceKoszulSetFormDegree(*koszulsp,formDegree);CHKERRQ(ierr);

  ierr = PetscSpaceKoszulSetDomain(*koszulsp,*domainsp);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt((*domainsp)->degree,formDegree,&nChooseK);CHKERRQ(ierr);
  if (Nc % nChooseK != 0){
      SETERRQ(PETSC_COMM_SELF,62,"Requested number of components is incompatible with given domain space and form degree.\n");
  }
  ierr = PetscSpaceSetNumComponents(*koszulsp,Nc);CHKERRQ(ierr);
  kosz = (PetscSpace_Koszul*) (*koszulsp)->data;

  kosz->setupCalled = 1;


  PetscFunctionReturn(0);
}
