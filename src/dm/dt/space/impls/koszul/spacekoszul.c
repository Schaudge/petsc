#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

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
  ierr     = PetscSpaceInitialize_Sum(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreateKoszul(PetscSpace *domainsp,PetscSpace *koszulsp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
