
#include <petsc/private/matimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/dmimpl.h>
#include <petscksp.h>           /*I "petscksp.h" I*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscsnes.h>
#include <petsc/private/kspimpl.h>
#include <petsc/private/snesimpl.h>

#include "../src/ksp/pc/impls/telescope/telescope.h"

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@inproceedings{MaySananRuppKnepleySmith2016,\n"
"  title     = {Extreme-Scale Multigrid Components within PETSc},\n"
"  author    = {Dave A. May and Patrick Sanan and Karl Rupp and Matthew G. Knepley and Barry F. Smith},\n"
"  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},\n"
"  series    = {PASC '16},\n"
"  isbn      = {978-1-4503-4126-4},\n"
"  location  = {Lausanne, Switzerland},\n"
"  pages     = {5:1--5:12},\n"
"  articleno = {5},\n"
"  numpages  = {12},\n"
"  url       = {http://doi.acm.org/10.1145/2929908.2929913},\n"
"  doi       = {10.1145/2929908.2929913},\n"
"  acmid     = {2929913},\n"
"  publisher = {ACM},\n"
"  address   = {New York, NY, USA},\n"
"  keywords  = {GPU, HPC, agglomeration, coarse-level solver, multigrid, parallel computing, preconditioning},\n"
"  year      = {2016}\n"
"}\n";

typedef struct {
  DM              dm_fine,dm_coarse; /* these DM's should be topologically identical but use different communicators */
  Mat             permutation;
  Vec             xp;
  PetscErrorCode  (*fp_dm_field_scatter)(DM,Vec,ScatterMode,DM,Vec);
  PetscErrorCode  (*fp_dm_state_scatter)(DM,ScatterMode,DM);
  void            *dmksp_context_determined;
  void            *dmksp_context_user;
  PetscBool       has_dmsnes;
  SNES            snes_coarse;
  Vec             x_state_coarse;
} PC_Telescope_CoarseDMCtx;

PetscErrorCode PCTelescopeMatCreate_CoarseDM_DMSNES(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A);

PetscErrorCode PCTelescopeSetUp_scatters_CoarseDM(PC pc,PC_Telescope sred,PC_Telescope_CoarseDMCtx *ctx)
{
  PetscErrorCode ierr;
  Vec            xred,yred,xtmp,x,xp;
  VecScatter     scatter;
  IS             isin;
  Mat            B;
  PetscInt       m,bs,st,ed;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);
  ierr = MatCreateVecs(B,&x,NULL);CHKERRQ(ierr);
  ierr = MatGetBlockSize(B,&bs);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xp);CHKERRQ(ierr);
  m = 0;
  xred = NULL;
  yred = NULL;
  if (isActiveRank(sred)) {
    ierr = DMCreateGlobalVector(ctx->dm_coarse,&xred);CHKERRQ(ierr);
    ierr = VecDuplicate(xred,&yred);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(xred,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,ed-st,st,1,&isin);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xred,&m);CHKERRQ(ierr);
  } else {
    ierr = VecGetOwnershipRange(x,&st,&ed);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,0,st,1,&isin);CHKERRQ(ierr);
  }
  ierr = ISSetBlockSize(isin,bs);CHKERRQ(ierr);
  ierr = VecCreate(comm,&xtmp);CHKERRQ(ierr);
  ierr = VecSetSizes(xtmp,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(xtmp,bs);CHKERRQ(ierr);
  ierr = VecSetType(xtmp,((PetscObject)x)->type_name);CHKERRQ(ierr);
  ierr = VecScatterCreateWithData(x,isin,xtmp,NULL,&scatter);CHKERRQ(ierr);
  sred->xred    = xred;
  sred->yred    = yred;
  sred->isin    = isin;
  sred->scatter = scatter;
  sred->xtmp    = xtmp;

  ctx->xp       = xp;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSetUp_CoarseDM(PC pc,PC_Telescope sred)
{
  PC_Telescope_CoarseDMCtx *ctx;
  DM                 dm,dm_coarse = NULL;
  MPI_Comm           comm;
  PetscBool          has_perm,has_kspcomputeoperators;
  PetscBool          has_dmksp,has_dmsnes;
  DMKSP              dmk = NULL;
  DMSNES             dms = NULL;
  PetscErrorCode     (*dmfine_kspfunc)(KSP,Mat,Mat,void*) = NULL;
  void               *dmfine_kspctx = NULL,*dmcoarse_kspctx = NULL;
  void               *dmfine_appctx = NULL,*dmcoarse_appctx = NULL;
  void               *dmfine_shellctx = NULL,*dmcoarse_shellctx = NULL;
  int                valid_setup = 0;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(pc,"PCTelescope: setup (CoarseDM)\n");CHKERRQ(ierr);
  ierr = PetscMalloc1(1,&ctx);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx,sizeof(PC_Telescope_CoarseDMCtx));CHKERRQ(ierr);
  sred->dm_ctx = (void*)ctx;

  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);

  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  ierr = DMGetCoarseDM(dm,&dm_coarse);CHKERRQ(ierr);
  ctx->dm_fine   = dm;
  ctx->dm_coarse = dm_coarse;

  /* attach coarse dm to ksp on sub communicator */
  if (isActiveRank(sred)) {
    ierr = KSPSetDM(sred->ksp,ctx->dm_coarse);CHKERRQ(ierr);
    if (sred->ignore_kspcomputeoperators) {
      ierr = KSPSetDMActive(sred->ksp,PETSC_FALSE);CHKERRQ(ierr);
    }
  }

  /* check if there is a method to provide a permutation */
  has_perm                = PETSC_FALSE;
  has_kspcomputeoperators = PETSC_FALSE;
  has_dmsnes              = PETSC_FALSE;
  has_dmksp               = PETSC_FALSE;
  valid_setup             = 0;

  ierr = DMKSPGetComputeOperators(dm,&dmfine_kspfunc,&dmfine_kspctx);CHKERRQ(ierr);
  if (dmfine_kspfunc) { has_kspcomputeoperators = PETSC_TRUE; }

  if (!has_perm && !has_kspcomputeoperators) SETERRQ(comm,PETSC_ERR_SUP,"No method to permute an operator was found on the parent DM. No method for KSPSetComputeOperators() was provided. Telescope setup cannot proceed");

  if (!has_perm && has_kspcomputeoperators && sred->ignore_kspcomputeoperators) SETERRQ(comm,PETSC_ERR_SUP,"No method to permute an operator was found on the parent DM. A method for KSPSetComputeOperators() was provided but it was requested to be ignored. Telescope setup cannot proceed");

  ierr = DMGetDMSNES(ctx->dm_fine,&dms);CHKERRQ(ierr);
  if (dms) { has_dmsnes = PETSC_TRUE; has_dmksp = PETSC_TRUE; ctx->has_dmsnes = PETSC_TRUE; }
  if (!has_dmsnes) {
    ierr = DMGetDMKSP(ctx->dm_fine,&dmk);CHKERRQ(ierr);
    if (dmk) { has_dmksp = PETSC_TRUE; }
  }

  ierr = DMGetApplicationContext(ctx->dm_fine,&dmfine_appctx);CHKERRQ(ierr);
  ierr = DMShellGetContext(ctx->dm_fine,&dmfine_shellctx);CHKERRQ(ierr);
  if (isActiveRank(sred)) {
    ierr = DMGetApplicationContext(ctx->dm_coarse,&dmcoarse_appctx);CHKERRQ(ierr);
    ierr = DMShellGetContext(ctx->dm_coarse,&dmcoarse_shellctx);CHKERRQ(ierr);
  }

  if (has_dmsnes) {
    SNES snes_coarse = NULL;

    if (isActiveRank(sred)) {
      /* create a new snes on sub-communicator */
      ierr = SNESCreate(PetscObjectComm((PetscObject)dm_coarse),&snes_coarse);CHKERRQ(ierr);
      ierr = SNESSetDM(snes_coarse,dm_coarse);CHKERRQ(ierr);
      ierr = DMCopyDMSNES(dm,dm_coarse);CHKERRQ(ierr);
      /*ierr = DMGetDMSNES(dm_coarse,&dmsnes_coarse);CHKERRQ(ierr);*/
      /*ierr = DMSNESSetFunction(dm_coarse,NULL,(void*)snes_coarse);CHKERRQ(ierr);*/
      ctx->dmksp_context_user = (void*)snes_coarse;
      dmcoarse_kspctx = (void*)snes_coarse;
      valid_setup = 1;
      ctx->snes_coarse = snes_coarse;
    }
  } else if (!has_dmsnes && has_dmksp) {

    /* Assume that if the fine operator didn't require any context, neither will the coarse */
    if (!dmfine_kspctx) {
      dmcoarse_kspctx = NULL;
      ierr = PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using NULL context\n");CHKERRQ(ierr);
      valid_setup = 1;
    } else {
      ierr = PetscInfo(pc,"PCTelescope: KSPSetComputeOperators detected non-NULL context from parent DM \n");CHKERRQ(ierr);
      if (isActiveRank(sred)) {
        if (dmfine_kspctx == dmfine_appctx) {
          dmcoarse_kspctx = dmcoarse_appctx;
          valid_setup = 1;
          ierr = PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using context from DM->ApplicationContext\n");CHKERRQ(ierr);
          if (!dmcoarse_kspctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Non NULL dmfine->kspctx == dmfine->appctx. NULL dmcoarse->appctx found. Likely this is an error");
        } else if (dmfine_kspctx == dmfine_shellctx) {
          dmcoarse_kspctx = dmcoarse_shellctx;
          valid_setup = 1;
          ierr = PetscInfo(pc,"PCTelescope: KSPSetComputeOperators using context from DMShell->Context\n");CHKERRQ(ierr);
          if (!dmcoarse_kspctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Non NULL dmfine->kspctx == dmfine.shell->ctx. NULL dmcoarse.shell->ctx found. Likely this is an error");
        }
        ctx->dmksp_context_determined = dmcoarse_kspctx;
      }
    }

  } else {
    valid_setup = 0;
  }

  /* look for user provided method to fetch the context */
  {
    PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;
    void *dmcoarse_context_user = NULL;
    char dmcoarse_method[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMKSPContext");CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context);CHKERRQ(ierr);
    if (fp_get_coarsedm_context) {
      ierr = PetscInfo1(pc,"PCTelescope: Found composed method %s from coarse DM\n",dmcoarse_method);CHKERRQ(ierr);
      ierr = fp_get_coarsedm_context(ctx->dm_coarse,&dmcoarse_context_user);CHKERRQ(ierr);
      ctx->dmksp_context_user = dmcoarse_context_user;
      dmcoarse_kspctx = dmcoarse_context_user;
      valid_setup = 1;
    } else {
      ierr = PetscInfo1(pc,"PCTelescope: Failed to find composed method %s from coarse DM\n",dmcoarse_method);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Allreduce(MPI_IN_PLACE,&valid_setup,1,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
  if (valid_setup == 0) {
    ierr = PetscInfo(pc,"PCTelescope: KSPSetComputeOperators failed to determine the context to use on sub-communicator\n");CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot determine which context with use for KSPSetComputeOperators() on sub-communicator");
  }

  if (isActiveRank(sred)) {
    /* sub ksp inherits dmksp_func and context provided by user */
    ierr = KSPSetComputeOperators(sred->ksp,dmfine_kspfunc,dmcoarse_kspctx);CHKERRQ(ierr);
    /*ierr = PetscObjectCopyFortranFunctionPointers((PetscObject)dm,(PetscObject)ctx->dmrepart);CHKERRQ(ierr);*/
    ierr = KSPSetDMActive(sred->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }

  {
    char dmfine_method[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(dmfine_method,sizeof(dmfine_method),"PCTelescopeFieldScatter");CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)ctx->dm_fine,dmfine_method,&ctx->fp_dm_field_scatter);CHKERRQ(ierr);

    ierr = PetscSNPrintf(dmfine_method,sizeof(dmfine_method),"PCTelescopeStateScatter");CHKERRQ(ierr);
    ierr = PetscObjectQueryFunction((PetscObject)ctx->dm_fine,dmfine_method,&ctx->fp_dm_state_scatter);CHKERRQ(ierr);
  }

  if (ctx->fp_dm_state_scatter) {
    ierr = PetscInfo(pc,"PCTelescope: Found composed method PCTelescopeStateScatter from parent DM\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(pc,"PCTelescope: Failed to find composed method PCTelescopeStateScatter from parent DM\n");CHKERRQ(ierr);
  }

  if (ctx->fp_dm_field_scatter) {
    ierr = PetscInfo(pc,"PCTelescope: Found composed method PCTelescopeFieldScatter from parent DM\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(pc,"PCTelescope: Failed to find composed method PCTelescopeFieldScatter from parent DM\n");CHKERRQ(ierr);
    SETERRQ(comm,PETSC_ERR_SUP,"No method to scatter fields between the parent DM and coarse DM was found. Must call PetscObjectComposeFunction() with the parent DM. Telescope setup cannot proceed");
  }

  /*ierr = PCTelescopeSetUp_permutation_CoarseDM(pc,sred,ctx);CHKERRQ(ierr);*/

  ierr = PCTelescopeSetUp_scatters_CoarseDM(pc,sred,ctx);CHKERRQ(ierr);
  if (ctx->has_dmsnes) {
    SNES snes_coarse = NULL;
    Vec xcoarse = NULL;

    if (isActiveRank(sred)) {
      snes_coarse = ctx->snes_coarse;
      ierr = VecDuplicate(sred->xred,&xcoarse);CHKERRQ(ierr);
      ierr = SNESSetSolution(snes_coarse,xcoarse);CHKERRQ(ierr);
      ctx->x_state_coarse = xcoarse;
    }
  }
#if 0 /* this is executed in PCTelescopeMatCreate_CoarseDM() */
  if (ctx->has_dmsnes) {
    SNES snes_fine = NULL;
    SNES snes_coarse = NULL;
    Vec xfine = NULL;
    Vec xcoarse = NULL;

    ierr = DMKSPGetComputeOperators(ctx->dm_fine,NULL,(void**)&snes_fine);CHKERRQ(ierr);
    /* check the context really is a SNES object, it should be if the method is KSPComputeOperators_SNES() */
    PetscValidHeaderSpecific(snes_fine,SNES_CLASSID,3);
    ierr = SNESGetSolution(snes_fine,&xfine);CHKERRQ(ierr);
    if (isActiveRank(sred)) {
      snes_coarse = ctx->snes_coarse;
      ierr = SNESGetSolution(snes_coarse,&xcoarse);CHKERRQ(ierr);
    }
    if (xfine) {
      ierr = ctx->fp_dm_field_scatter(ctx->dm_fine,xfine,SCATTER_FORWARD,ctx->dm_coarse,xcoarse);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PCApply_Telescope_CoarseDM(PC pc,Vec x,Vec y)
{
  PC_Telescope      sred = (PC_Telescope)pc->data;
  PetscErrorCode    ierr;
  Vec               xred,yred;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  xred    = sred->xred;
  yred    = sred->yred;

  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);
  /*
  if (ctx->fp_dm_state_scatter) {
    ierr = ctx->fp_dm_state_scatter(ctx->dm_fine,SCATTER_FORWARD,ctx->dm_coarse);CHKERRQ(ierr);
  }
  */
  ierr = ctx->fp_dm_field_scatter(ctx->dm_fine,x,SCATTER_FORWARD,ctx->dm_coarse,xred);CHKERRQ(ierr);

  /* solve */
  if (isActiveRank(sred)) {
    ierr = KSPSolve(sred->ksp,xred,yred);CHKERRQ(ierr);
  }

  ierr = ctx->fp_dm_field_scatter(ctx->dm_fine,y,SCATTER_REVERSE,ctx->dm_coarse,yred);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeSubNullSpaceCreate_CoarseDM(PC pc,PC_Telescope sred,MatNullSpace nullspace,MatNullSpace *sub_nullspace)
{
  PetscErrorCode   ierr;
  PetscBool        has_const;
  PetscInt         k,n = 0;
  const Vec        *vecs;
  Vec              *sub_vecs = NULL;
  MPI_Comm         subcomm;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  subcomm = sred->subcomm;
  ierr = MatNullSpaceGetVecs(nullspace,&has_const,&n,&vecs);CHKERRQ(ierr);

  if (isActiveRank(sred)) {
    /* create new vectors */
    if (n) {
      ierr = VecDuplicateVecs(sred->xred,n,&sub_vecs);CHKERRQ(ierr);
    }
  }

  /* copy entries */
  for (k=0; k<n; k++) {
    ierr = ctx->fp_dm_field_scatter(ctx->dm_fine,vecs[k],SCATTER_FORWARD,ctx->dm_coarse,sub_vecs[k]);CHKERRQ(ierr);
  }

  if (isActiveRank(sred)) {
    /* create new (near) nullspace for redundant object */
    ierr = MatNullSpaceCreate(subcomm,has_const,n,sub_vecs,sub_nullspace);CHKERRQ(ierr);
    ierr = VecDestroyVecs(n,&sub_vecs);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatNullSpaceCreate_CoarseDM(PC pc,PC_Telescope sred,Mat sub_mat)
{
  PetscErrorCode   ierr;
  Mat              B;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  ierr = PCGetOperators(pc,NULL,&B);CHKERRQ(ierr);

  {
    MatNullSpace nullspace,sub_nullspace;
    ierr = MatGetNullSpace(B,&nullspace);CHKERRQ(ierr);
    if (nullspace) {
      ierr = PetscInfo(pc,"PCTelescope: generating nullspace (CoarseDM)\n");CHKERRQ(ierr);
      ierr = PCTelescopeSubNullSpaceCreate_CoarseDM(pc,sred,nullspace,&sub_nullspace);CHKERRQ(ierr);

      /* attach any user nullspace removal methods and contexts */
      if (isActiveRank(sred)) {
        void *context = NULL;
        if (nullspace->remove && !nullspace->rmctx){
          ierr = MatNullSpaceSetFunction(sub_nullspace,nullspace->remove,context);CHKERRQ(ierr);
        } else if (nullspace->remove && nullspace->rmctx) {
          char dmcoarse_method[PETSC_MAX_PATH_LEN];
          PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;

          ierr = PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMNullSpaceUserContext");CHKERRQ(ierr);
          ierr = PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context);CHKERRQ(ierr);

          if (!context) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Propagation of user null-space removal method with non-NULL context requires the coarse DM be composed with a function named \"%s\"",dmcoarse_method);

          ierr = MatNullSpaceSetFunction(sub_nullspace,nullspace->remove,context);CHKERRQ(ierr);
        }
      }

      if (isActiveRank(sred)) {
        ierr = MatSetNullSpace(sub_mat,sub_nullspace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&sub_nullspace);CHKERRQ(ierr);
      }
    }
  }

  {
    MatNullSpace nearnullspace,sub_nearnullspace;
    ierr = MatGetNearNullSpace(B,&nearnullspace);CHKERRQ(ierr);
    if (nearnullspace) {
      ierr = PetscInfo(pc,"PCTelescope: generating near nullspace (CoarseDM)\n");CHKERRQ(ierr);
      ierr = PCTelescopeSubNullSpaceCreate_CoarseDM(pc,sred,nearnullspace,&sub_nearnullspace);CHKERRQ(ierr);

      /* attach any user nullspace removal methods and contexts */
      if (isActiveRank(sred)) {
        void *context = NULL;
        if (nearnullspace->remove && !nearnullspace->rmctx){
          ierr = MatNullSpaceSetFunction(sub_nearnullspace,nearnullspace->remove,context);CHKERRQ(ierr);
        } else if (nearnullspace->remove && nearnullspace->rmctx) {
          char dmcoarse_method[PETSC_MAX_PATH_LEN];
          PetscErrorCode (*fp_get_coarsedm_context)(DM,void**) = NULL;

          ierr = PetscSNPrintf(dmcoarse_method,sizeof(dmcoarse_method),"PCTelescopeGetCoarseDMNearNullSpaceUserContext");CHKERRQ(ierr);
          ierr = PetscObjectQueryFunction((PetscObject)ctx->dm_coarse,dmcoarse_method,&fp_get_coarsedm_context);CHKERRQ(ierr);

          if (!context) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Propagation of user near null-space removal method with non-NULL context requires the coarse DM be composed with a function named \"%s\"",dmcoarse_method);

          ierr = MatNullSpaceSetFunction(sub_nearnullspace,nearnullspace->remove,context);CHKERRQ(ierr);
        }
      }

      if (isActiveRank(sred)) {
        ierr = MatSetNearNullSpace(sub_mat,sub_nearnullspace);CHKERRQ(ierr);
        ierr = MatNullSpaceDestroy(&sub_nearnullspace);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_Telescope_CoarseDM(PC pc)
{
  PetscErrorCode       ierr;
  PC_Telescope         sred = (PC_Telescope)pc->data;
  PC_Telescope_CoarseDMCtx *ctx;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  ctx->dm_fine = NULL; /* since I did not increment the ref counter we set these to NULL */
  ctx->dm_coarse = NULL; /* since I did not increment the ref counter we set these to NULL */
  ctx->permutation = NULL; /* this will be fetched from the dm so no need to call destroy */
  ierr = VecDestroy(&ctx->xp);CHKERRQ(ierr);
  ctx->fp_dm_field_scatter = NULL;
  ctx->fp_dm_state_scatter = NULL;
  ctx->dmksp_context_determined = NULL;
  ctx->dmksp_context_user = NULL;
  ierr = SNESDestroy(&ctx->snes_coarse);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x_state_coarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCApplyRichardson_Telescope_CoarseDM(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt its,PetscBool zeroguess,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_Telescope      sred = (PC_Telescope)pc->data;
  PetscErrorCode    ierr;
  Vec               yred = NULL;
  PetscBool         default_init_guess_value = PETSC_FALSE;
  PC_Telescope_CoarseDMCtx *ctx;

  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  yred    = sred->yred;

  if (its > 1) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCApplyRichardson_Telescope_CoarseDM only supports max_it = 1");
  *reason = (PCRichardsonConvergedReason)0;
  
  if (!zeroguess) {
    ierr = PetscInfo(pc,"PCTelescopeCoarseDM: Scattering y for non-zero-initial guess\n");CHKERRQ(ierr);

    ierr = ctx->fp_dm_field_scatter(ctx->dm_fine,y,SCATTER_FORWARD,ctx->dm_coarse,yred);CHKERRQ(ierr);
  }

  if (isActiveRank(sred)) {
    ierr = KSPGetInitialGuessNonzero(sred->ksp,&default_init_guess_value);CHKERRQ(ierr);
    if (!zeroguess) ierr = KSPSetInitialGuessNonzero(sred->ksp,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PCApply_Telescope_CoarseDM(pc,x,y);CHKERRQ(ierr);

  if (isActiveRank(sred)) {
    ierr = KSPSetInitialGuessNonzero(sred->ksp,default_init_guess_value);CHKERRQ(ierr);
  }

  if (!*reason) *reason = PCRICHARDSON_CONVERGED_ITS;
  *outits = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_CoarseDM_Default(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  DM             dm_fine,dm_coarse;
  PC_Telescope_CoarseDMCtx *ctx = NULL;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  dm_fine = ctx->dm_fine;
  dm_coarse = ctx->dm_coarse;
  if (reuse == MAT_INITIAL_MATRIX) {
    /*
     There is no need to explicitly create the operator now,
     the sub-KSP will do this automatically during KSPSetUp()
     */
    if (ctx->fp_dm_state_scatter) {
      ierr = ctx->fp_dm_state_scatter(dm_fine,SCATTER_FORWARD,dm_coarse);CHKERRQ(ierr);
    }
  } else if (reuse == MAT_REUSE_MATRIX) {
    if (ctx->fp_dm_state_scatter) {
      ierr = ctx->fp_dm_state_scatter(dm_fine,SCATTER_FORWARD,dm_coarse);CHKERRQ(ierr);
    }
  }
  /*
   There is no need to explicitly assemble the operator now,
   the sub-KSP will call the method provided to KSPSetComputeOperators() during KSPSetUp()
   */
  PetscFunctionReturn(0);
}

/*
 Defines the special method to setup the operator if the DM has an attached DMSNES
 */
PetscErrorCode PCTelescopeMatCreate_CoarseDM_DMSNES(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  DM             dm_fine,dm_coarse;
  Vec            x_state_fine = NULL, x_state_coarse = NULL;
  SNES           snes_fine = NULL,snes_coarse = NULL;
  PC_Telescope_CoarseDMCtx *ctx = NULL;

  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  dm_fine = ctx->dm_fine;
  dm_coarse = ctx->dm_coarse;
  ierr = DMKSPGetComputeOperators(dm_fine,NULL,&snes_fine);CHKERRQ(ierr);
  PetscValidHeaderSpecific(snes_fine,SNES_CLASSID,3);
  ierr = SNESGetSolution(snes_fine,&x_state_fine);CHKERRQ(ierr);

  if (isActiveRank(sred)) {
    ierr = DMKSPGetComputeOperators(dm_coarse,NULL,&snes_coarse);CHKERRQ(ierr);
    PetscValidHeaderSpecific(snes_coarse,SNES_CLASSID,3);
    ierr = SNESGetSolution(snes_coarse,&x_state_coarse);CHKERRQ(ierr);
  }

  /*
   The stored state on the DM might depend on the SNES iterate x, so we
   scatter the non-linear state vector first.
   */
  if (reuse == MAT_INITIAL_MATRIX) {
    /*
     There is no need to explicitly create the operator now,
     the sub-KSP will do this automatically during KSPSetUp()
     */
    ierr = ctx->fp_dm_field_scatter(dm_fine,x_state_fine,SCATTER_FORWARD,dm_coarse,x_state_coarse);CHKERRQ(ierr);

    if (ctx->fp_dm_state_scatter) {
      ierr = ctx->fp_dm_state_scatter(dm_fine,SCATTER_FORWARD,dm_coarse);CHKERRQ(ierr);
    }
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = ctx->fp_dm_field_scatter(dm_fine,x_state_fine,SCATTER_FORWARD,dm_coarse,x_state_coarse);CHKERRQ(ierr);
    if (ctx->fp_dm_state_scatter) {
      ierr = ctx->fp_dm_state_scatter(dm_fine,SCATTER_FORWARD,dm_coarse);CHKERRQ(ierr);
    }
  }
  /*
   There is no need to explicitly assemble the operator now,
   the sub-KSP will call the method provided to KSPSetComputeOperators() during KSPSetUp()
   */
  PetscFunctionReturn(0);
}

PetscErrorCode PCTelescopeMatCreate_CoarseDM(PC pc,PC_Telescope sred,MatReuse reuse,Mat *A)
{
  PetscErrorCode ierr;
  PC_Telescope_CoarseDMCtx *ctx = NULL;
  PetscFunctionBegin;
  ctx = (PC_Telescope_CoarseDMCtx*)sred->dm_ctx;
  if (ctx->has_dmsnes) {
    ierr = PCTelescopeMatCreate_CoarseDM_DMSNES(pc,sred,reuse,A);CHKERRQ(ierr);
  } else {
    ierr = PCTelescopeMatCreate_CoarseDM_Default(pc,sred,reuse,A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
