#include <../src/tao/constrained/impls/mad/mad.h> /*I "petsctao.h" I*/ /*I "petscsnes.h" I*/ /*I "petscvec.h" I*/ /*I "petscblaslapack.h" I*/

static PetscErrorCode TaoMADSNESCheck(SNES snes, PetscInt it, PetscReal xnorm, PetscReal dnorm, PetscReal gnorm, SNESConvergedReason* reason, void* ctx)
{
  Tao tao = (Tao)ctx;
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->niter = it;
  ierr = TaoMADCheckConvergence(tao, mad->Q, mad->L, mad->dLdQ, mad->alpha);CHKERRQ(ierr);
  TaoMADConvertReasonToSNES(tao->reason, reason);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMADSNESFunction(SNES snes, Vec Rsnes, Vec Gsnes, void* ctx)
{
  Tao tao = (Tao)ctx;
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* NOTE: X is the solution component of mad->Q */
  /* NOTE: F is the reduced gradient component of mad->G */
  /* NOTE: we can assume that the full-space gradient dLdQ has already been computed */
  /* save the current solution for reverting later */
  /* update barrier and evaluate reduced KKT */
  ierr = VecCopy(Rsnes, mad->Q->R);CHKERRQ(ierr);
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Q, mad->L, mad->dLdQ);CHKERRQ(ierr);
  ierr = TaoMADComputeReducedKKT(tao, mad->Q, mad->dLdQ, mad->G);CHKERRQ(ierr);
  ierr = VecCopy(mad->G->R, Gsnes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMADSNESLineSearchPreCheck(SNESLineSearch linesearch, Vec X, Vec D, PetscBool* changed, void* ctx)
{
  Tao tao = (Tao)ctx;
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(D, mad->D->R);CHKERRQ(ierr);
  ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
  *changed = PETSC_FALSE;
  if (mad->use_ipm) {
    ierr = TaoMADEvaluateClosedFormUpdates(tao, mad->Q, mad->dLdQ, mad->D);CHKERRQ(ierr);
  } else {
    ierr = TaoMADEstimateActiveSet(tao, mad->Q->X, mad->dLdQ->X, mad->as_step, mad->D->X, changed);CHKERRQ(ierr);
    /* adjust step for bounds and take zero step for inactive multipliers */
    if (*changed) {
      if (tao->bounded) {
        ierr = TaoBoundStep(mad->Q->X, tao->XL, tao->XU, mad->isXL, mad->isXU, mad->fixedXB, 1.0, mad->D->X);CHKERRQ(ierr);
      }
      if (tao->ineq_constrained) {
        ierr = VecISSet(mad->D->Yi, mad->inactiveCI, 0.0);CHKERRQ(ierr);
      }
      ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
      ierr = VecCopy(mad->D->R, D);CHKERRQ(ierr);
      ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMADSNESLineSearchPostCheck(SNESLineSearch linesearch, Vec X, Vec D, Vec W, PetscBool* changed_D, PetscBool* changed_W, void* ctx)
{
  Tao tao = (Tao)ctx;
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  SNESLineSearchReason reason;
  PetscReal alpha;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *changed_D = PETSC_FALSE;
  *changed_W = PETSC_FALSE;
  ierr = SNESLineSearchGetReason(linesearch, &reason);CHKERRQ(ierr);
  if (reason != SNES_LINESEARCH_SUCCEEDED) PetscFunctionReturn(0);
  /* update TAO step with SNES step length */
  ierr = SNESLineSearchGetLambda(linesearch, &alpha);CHKERRQ(ierr);
  ierr = VecAXPY(mad->Q->F, alpha, mad->D->F);CHKERRQ(ierr);
  if (mad->use_ipm) {
    ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);
  }
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Q, mad->L, mad->dLdQ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMADSNESLineSearchApply(SNESLineSearch linesearch, void* ctx)
{
  Tao tao = (Tao)ctx;
  TAO_MAD* mad = (TAO_MAD*)tao->data;
  TaoLineSearchConvergedReason reason;
  Vec X, G, D, Wx, Wg;
  PetscBool changed, success;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESLineSearchGetVecs(linesearch, &X, &G, &D, &Wx, &Wg);CHKERRQ(ierr);
  ierr = VecCopy(X, mad->Q->R);CHKERRQ(ierr);
  ierr = VecCopy(G, mad->dLdQ->R);CHKERRQ(ierr);
  ierr = VecCopy(D, mad->D->R);CHKERRQ(ierr);
  ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);

  /* apply step corrections */
  if (mad->use_ipm) {
    /* IPM needs missing updates computed */
    ierr = TaoMADEvaluateClosedFormUpdates(tao, mad->Q, mad->dLdQ, mad->D);CHKERRQ(ierr);
  } else {
    /* active-set step needs to be adjusted for bounds */
    ierr = TaoMADEstimateActiveSet(tao, mad->Q->X, mad->dLdQ->X, mad->as_step, mad->D->X, &changed);CHKERRQ(ierr);
    if (changed) {
      if (tao->bounded) {
        ierr = TaoBoundStep(mad->Q->X, tao->XL, tao->XU, mad->isXL, mad->isXU, mad->fixedXB, 1.0, mad->D->X);CHKERRQ(ierr);
      }
      if (tao->ineq_constrained) {
        ierr = VecISSet(mad->D->Yi, mad->inactiveCI, 0.0);CHKERRQ(ierr);
      }
      ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
      ierr = VecCopy(mad->D->R, D);CHKERRQ(ierr);
      ierr = VecScale(mad->D->R, -1.0);CHKERRQ(ierr);
    }
  }

  /* preserve starting point for a potential restart */
  ierr = VecCopy(mad->Q->F, mad->Qprev->F);CHKERRQ(ierr);
  ierr = VecCopy(mad->dLdQ->F, mad->dLdQprev->F);CHKERRQ(ierr);
  ierr = LagrangianCopy(mad->L, mad->Lprev);CHKERRQ(ierr);

  success = PETSC_FALSE;
  if (mad->use_filter) {
    /* apply step with safeguard */
    ierr = TaoMADApplyFilterStep(tao, mad->Q, mad->D, mad->L, mad->dLdQ, &mad->alpha);CHKERRQ(ierr);
    if (mad->alpha == 0.0) {
      /* step failed, reset to initial point */
      mad->restarts += 1;
      ierr = VecCopy(mad->Qprev->F, mad->Q->F);CHKERRQ(ierr);
      ierr = VecCopy(mad->dLdQprev->F, mad->dLdQ->F);CHKERRQ(ierr);
      ierr = LagrangianCopy(mad->Lprev, mad->L);CHKERRQ(ierr);
      /* try again with gradient descent */
      ierr = VecCopy(mad->dLdQ->F, mad->D->F);CHKERRQ(ierr);
      ierr = VecScale(mad->D->F, -1.0);CHKERRQ(ierr);
      ierr = TaoMADApplyFilterStep(tao, mad->Q, mad->D, mad->L, mad->dLdQ, &mad->alpha);CHKERRQ(ierr);
      if (mad->alpha != 0.0) {
        /* gradient descent worked, reset MAD history */
        success = PETSC_TRUE;
        ierr = SNESNGMRESManualRestart(mad->snes);CHKERRQ(ierr);
      }
    } else success = PETSC_TRUE;
  } else {
    ierr = TaoLineSearchApply(tao->linesearch, mad->Q->R, &mad->L->val, mad->dLdQ->R, mad->D->R, &mad->alpha, &reason);CHKERRQ(ierr);
    if (reason != TAOLINESEARCH_SUCCESS) {
      /* step failed, reset to initial point */
      mad->restarts += 1;
      ierr = VecCopy(mad->Qprev->F, mad->Q->F);CHKERRQ(ierr);
      ierr = VecCopy(mad->dLdQprev->F, mad->dLdQ->F);CHKERRQ(ierr);
      ierr = LagrangianCopy(mad->Lprev, mad->L);CHKERRQ(ierr);
      /* try again with gradient descent */
      ierr = VecCopy(mad->dLdQ->F, mad->D->F);CHKERRQ(ierr);
      ierr = VecScale(mad->D->F, -1.0);CHKERRQ(ierr);
      if (tao->bounded) {
        ierr = TaoBoundStep(mad->Q->X, tao->XL, tao->XU, mad->isXL, mad->isXU, mad->fixedXB, 1.0, mad->D->X);CHKERRQ(ierr);
      }
      if (tao->ineq_constrained) {
        ierr = VecISSet(mad->D->Yi, mad->inactiveCI, 0.0);CHKERRQ(ierr);
      }
      ierr = TaoLineSearchApply(tao->linesearch, mad->Q->R, &mad->L->val, mad->dLdQ->R, mad->D->R, &mad->alpha, &reason);CHKERRQ(ierr);
      if (reason != TAOLINESEARCH_SUCCESS) {
        /* gradient descent worked, reset MAD history */
        success = PETSC_TRUE;
        ierr = SNESNGMRESManualRestart(mad->snes);CHKERRQ(ierr);
      }
    } else success = PETSC_TRUE;
  }

  if (success) {
    ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);
    ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);
    ierr = VecCopy(mad->Q->R, X);CHKERRQ(ierr);
    ierr = VecCopy(mad->dLdQ->R, G);CHKERRQ(ierr);
    ierr = SNESLineSearchComputeNorms(linesearch);CHKERRQ(ierr);
  } else {
    ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_FAILED_REDUCT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoMADLineSearchObjAndGrad(TaoLineSearch ls, Vec X, PetscReal *f, Vec G, void* ctx)
{
  Tao                 tao = (Tao)ctx;
  TAO_MAD             *mad = (TAO_MAD*)tao->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = VecCopy(X, mad->Qwork->R);CHKERRQ(ierr);
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Qwork, mad->Lwork, mad->dLdQwork);CHKERRQ(ierr);
  ierr = TaoMADComputeReducedKKT(tao, mad->Qwork, mad->dLdQwork, mad->G);CHKERRQ(ierr);
  *f = mad->Lwork->val;
  ierr = VecCopy(mad->dLdQwork->R, G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSolve_MAD(Tao tao)
{
  TAO_MAD             *mad = (TAO_MAD*)tao->data;
  SNESConvergedReason snes_reason;
  TaoConvergedReason  tao_reason;
  Mat                 AeT, AiT;
  Vec                 Xb, Cb;
  PetscReal           ginf;
  PetscScalar         *Ae_norms, *Ai_norms;
  PetscInt            nDiff;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* snap variables to bounds and set initial multipliers */
  if (tao->bounded) {
    ierr = TaoBoundSolution(mad->Q->X, tao->XL, tao->XU, 0.0, &nDiff, mad->Q->X);CHKERRQ(ierr);
  }
  if (mad->Q->Y) {
    ierr = VecSet(mad->Q->Y, 0.0);CHKERRQ(ierr);
  }

  /* initial slacks are a safeguarded clone of the corresponding constraints (Vanderbei and Shanno, 1998) */
  /* however, in their formulation their slacks have a negative sign, ours is positive, so we flip */
  if (mad->use_ipm) {
    if (tao->ineq_constrained) {
      ierr = TaoComputeInequalityConstraints(tao, mad->Q->X, mad->Ci);CHKERRQ(ierr);
      ierr = VecSet(mad->W->Sc, mad->slack_init);CHKERRQ(ierr);
      ierr = VecPointwiseMax(mad->Q->Sc, mad->Ci, mad->W->Sc);CHKERRQ(ierr);
      if (mad->isIL) {
        ierr = VecGetSubVector(mad->Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
        ierr = VecWAXPY(mad->Q->Scl, -1.0, Cb, mad->IL);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Q->Sc, mad->isIL, &Cb);CHKERRQ(ierr);
        ierr = VecSet(mad->W->Scl, mad->slack_init);CHKERRQ(ierr);
        ierr = VecPointwiseMax(mad->Q->Scl, mad->Q->Scl, mad->W->Scl);CHKERRQ(ierr);
      }
      if (mad->isIU) {
        ierr = VecGetSubVector(mad->Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
        ierr = VecWAXPY(mad->Q->Scu, -1.0, mad->IU, Cb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Q->Sc, mad->isIU, &Cb);CHKERRQ(ierr);
        ierr = VecSet(mad->W->Scu, mad->slack_init);CHKERRQ(ierr);
        ierr = VecPointwiseMax(mad->Q->Scu, mad->Q->Scu, mad->W->Scu);CHKERRQ(ierr);
      }
    }
    if (tao->bounded) {
      if (mad->isXL) {
        ierr = VecGetSubVector(mad->Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
        ierr = VecWAXPY(mad->Q->Sxl, -1.0, Xb, mad->XL);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Q->X, mad->isXL, &Xb);CHKERRQ(ierr);
        ierr = VecSet(mad->W->Sxl, mad->slack_init);CHKERRQ(ierr);
        ierr = VecPointwiseMax(mad->Q->Sxl, mad->Q->Sxl, mad->W->Sxl);CHKERRQ(ierr);
      }
      if (mad->isXU) {
        ierr = VecGetSubVector(mad->Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
        ierr = VecWAXPY(mad->Q->Sxu, -1.0, mad->XU, Xb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Q->X, mad->isXU, &Xb);CHKERRQ(ierr);
        ierr = VecSet(mad->W->Sxu, mad->slack_init);CHKERRQ(ierr);
        ierr = VecPointwiseMax(mad->Q->Sxu, mad->Q->Sxu, mad->W->Sxu);CHKERRQ(ierr);
      }
    }
    /* since we have initial multipliers and slacks, we can now estimate the barrier parameter */
    ierr = TaoMADUpdateBarrier(tao, mad->Q, &mad->mu);CHKERRQ(ierr);
  }

  /* compute primal and dual scaling factors (from Wachter and Biegler, 2004 [IPOPT]) */
  ierr = TaoMADComputeLagrangianAndGradient(tao, mad->Q, mad->L, mad->dLdQ);CHKERRQ(ierr);
  ierr = VecNorm(mad->dLdQ->P, NORM_INFINITY, &ginf);CHKERRQ(ierr);
  mad->Gscale = PetscMin(1.0, mad->scale_max/ginf);
  if (tao->eq_constrained) {
    ierr = MatTranspose(mad->Ae, MAT_INITIAL_MATRIX, &AeT);CHKERRQ(ierr);
    ierr = VecGetArray(mad->W->Ye, &Ae_norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(AeT, NORM_INFINITY, Ae_norms);CHKERRQ(ierr);
    ierr = VecRestoreArray(mad->W->Ye, &Ae_norms);CHKERRQ(ierr);
    ierr = MatDestroy(&AeT);CHKERRQ(ierr);
    ierr = VecSet(mad->CeScale, mad->scale_max);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mad->CeScale, mad->CeScale, mad->W->Ye);CHKERRQ(ierr);
    ierr = VecSet(mad->W->Ye, 1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMin(mad->CeScale, mad->W->Ye, mad->CeScale);CHKERRQ(ierr);
  }
  if (tao->ineq_constrained) {
    ierr = MatTranspose(mad->Ai, MAT_INITIAL_MATRIX, &AiT);CHKERRQ(ierr);
    ierr = VecGetArray(mad->W->Yi, &Ai_norms);CHKERRQ(ierr);
    ierr = MatGetColumnNorms(AiT, NORM_INFINITY, Ai_norms);CHKERRQ(ierr);
    ierr = VecRestoreArray(mad->W->Yi, &Ai_norms);CHKERRQ(ierr);
    ierr = MatDestroy(&AiT);CHKERRQ(ierr);
    ierr = VecSet(mad->CiScale, mad->scale_max);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(mad->CiScale, mad->CiScale, mad->W->Yi);CHKERRQ(ierr);
    ierr = VecSet(mad->W->Yi, 1.0);CHKERRQ(ierr);
    ierr = VecPointwiseMin(mad->CiScale, mad->W->Yi, mad->CiScale);CHKERRQ(ierr);
    /* propogate Ci scale to bound values */
    if (tao->IL) {
      ierr = VecPointwiseMult(tao->IL, tao->IL, mad->CiScale);CHKERRQ(ierr);
      if (mad->use_ipm) {
        ierr = VecGetSubVector(mad->CiScale, mad->isIL, &Cb);CHKERRQ(ierr);
        ierr = VecPointwiseMult(mad->IL, mad->IL, Cb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->CiScale, mad->isIL, &Cb);CHKERRQ(ierr);
      }
    }
    if (tao->IU) {
      ierr = VecPointwiseMult(tao->IU, tao->IU, mad->CiScale);CHKERRQ(ierr);
      if (mad->use_ipm) {
        ierr = VecGetSubVector(mad->CiScale, mad->isIU, &Cb);CHKERRQ(ierr);
        ierr = VecPointwiseMult(mad->IU, mad->IU, Cb);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->CiScale, mad->isIU, &Cb);CHKERRQ(ierr);
      }
    }
  }

  ierr = VecCopy(mad->Q->R, mad->Rsnes);CHKERRQ(ierr);
  ierr = SNESSolve(mad->snes, NULL, mad->Rsnes);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(mad->snes, &snes_reason);CHKERRQ(ierr);
  TaoMADConvertReasonFromSNES(snes_reason, &tao_reason);
  ierr = TaoSetConvergedReason(tao, tao_reason);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_MAD(Tao tao,PetscViewer viewer)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "direction restarts = %D\n", mad->restarts);CHKERRQ(ierr);
    ierr = SNESView(mad->snes, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_MAD(Tao tao)
{
  TAO_MAD              *mad = (TAO_MAD*)tao->data;
  SNESLineSearch       linesearch;
  DM                   dm;
  Vec                  Wtmp;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) PetscFunctionReturn(0);
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);

  /* start with the basics...aliasing */
  mad->Q->X = tao->solution;
  ierr = VecGetSize(mad->Q->X, &mad->Nx);CHKERRQ(ierr);
  ierr = VecDuplicate(mad->Q->X, &mad->G->X);CHKERRQ(ierr);
  mad->G->nR=1;  mad->Q->nR=1;  mad->Q->nF=1;  mad->Q->nP=1;  mad->Q->nS=0;  mad->Q->nY=0;
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution, &tao->gradient);CHKERRQ(ierr);
  }
  mad->dFdX = tao->gradient;
  mad->Gscale = 1.0;

  /* check all the constraints types and create necessary vectors */
  if (tao->eq_constrained) {
    mad->unconstrained = PETSC_FALSE;
    mad->Ce = tao->constraints_equality;
    mad->Ae = tao->jacobian_equality;
    ierr = VecGetSize(mad->Ce, &mad->Ne);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->Q->Ye);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->G->Ye);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ce, &mad->CeScale);CHKERRQ(ierr);
    ierr = VecSet(mad->CeScale, 1.0);CHKERRQ(ierr);
    mad->G->nR += 1; mad->Q->nR += 1;  mad->Q->nF += 1;  mad->Q->nY += 1;
  }
  if (tao->ineq_constrained) {
    mad->unconstrained = PETSC_FALSE;
    mad->Ci = tao->constraints_inequality;
    mad->Ai = tao->jacobian_inequality;
    ierr = VecGetSize(mad->Ci, &mad->Ni);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->Q->Yi);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->G->Yi);CHKERRQ(ierr);
    ierr = VecDuplicate(mad->Ci, &mad->CiScale);CHKERRQ(ierr);
    ierr = VecSet(mad->CiScale, 1.0);CHKERRQ(ierr);
    mad->G->nR += 1;  mad->Q->nR += 1;  mad->Q->nY += 1;
    if (mad->use_ipm) {
      ierr = VecDuplicate(mad->Ci, &mad->Q->Sc);CHKERRQ(ierr);
      ierr = VecDuplicate(mad->Ci, &mad->B);CHKERRQ(ierr);
      mad->Q->nF += 1;  mad->Q->nP += 1;  mad->Q->nS +=1;
      if (!tao->ineq_doublesided) {
      /* user did not define lower/upper bounds so we assume c_i(x) >= 0 */
      ierr = VecDuplicate(mad->Ci, &tao->IL);CHKERRQ(ierr);
      ierr = VecSet(tao->IL, 0.0);CHKERRQ(ierr);
      }
      if (tao->IL) {
        ierr = VecSet(mad->Q->Yi, PETSC_NINFINITY);CHKERRQ(ierr);
        ierr = VecWhichGreaterThan(tao->IL, mad->Q->Yi, &mad->isIL);CHKERRQ(ierr);
        ierr = ISGetSize(mad->isIL, &mad->Ncl);CHKERRQ(ierr);
        ierr = VecGetSubVector(tao->IL, mad->isIL, &Wtmp);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Scl);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Vl);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->IL);CHKERRQ(ierr);
        ierr = VecCopy(Wtmp, mad->IL);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Ci, mad->isIL, &Wtmp);CHKERRQ(ierr);
        mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
      }
      if (tao->IU) {
        ierr = VecSet(mad->Q->Yi, PETSC_INFINITY);CHKERRQ(ierr);
        ierr = VecWhichGreaterThan(mad->Q->Yi, tao->IU, &mad->isIU);CHKERRQ(ierr);
        ierr = ISGetSize(mad->isIU, &mad->Ncu);CHKERRQ(ierr);
        ierr = VecGetSubVector(tao->IU, mad->isIU, &Wtmp);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Scu);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Vu);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->IU);CHKERRQ(ierr);
        ierr = VecCopy(Wtmp, mad->IU);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(mad->Ci, mad->isIU, &Wtmp);CHKERRQ(ierr);
        mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
      }
    } else {
      if (tao->ineq_doublesided) {
        SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "Double-sided inequality constraints are NOT supported by default active-set, switch to interior point with -tao_mad_use_ipm");
      } else {
        ierr = VecDuplicate(mad->Ci, &tao->IL);CHKERRQ(ierr);
        ierr = VecSet(tao->IL, 0.0);CHKERRQ(ierr);
        ierr = VecDuplicate(mad->Ci, &tao->IU);CHKERRQ(ierr);
        ierr = VecSet(tao->IU, PETSC_INFINITY);CHKERRQ(ierr);
        ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->inactiveCI);CHKERRQ(ierr);
      }
    }
  }
  if (tao->bounded) {
    mad->unconstrained = PETSC_FALSE;
    if (mad->use_ipm) {
      /* we need to determine index sets for bound constraints that are infinity */
      if (tao->XL) {
        ierr = VecSet(mad->dFdX, PETSC_NINFINITY);CHKERRQ(ierr);
        ierr = VecWhichGreaterThan(tao->XL, mad->dFdX, &mad->isXL);CHKERRQ(ierr);
        ierr = ISGetSize(mad->isXL, &mad->Nxl);CHKERRQ(ierr);
        ierr = VecGetSubVector(tao->XL, mad->isXL, &Wtmp);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Sxl);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Zl);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->XL);CHKERRQ(ierr);
        ierr = VecCopy(Wtmp, mad->XL);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(tao->solution, mad->isXL, &Wtmp);CHKERRQ(ierr);
        mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
      }
      if (tao->XU) {
        ierr = VecSet(mad->dFdX, PETSC_INFINITY);CHKERRQ(ierr);
        ierr = VecWhichGreaterThan(mad->dFdX, tao->XU, &mad->isXU);CHKERRQ(ierr);
        ierr = ISGetSize(mad->isXU, &mad->Nxu);CHKERRQ(ierr);
        ierr = VecGetSubVector(tao->XU, mad->isXU, &Wtmp);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Sxu);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->Q->Zu);CHKERRQ(ierr);
        ierr = VecDuplicate(Wtmp, &mad->XU);CHKERRQ(ierr);
        ierr = VecCopy(Wtmp, mad->XU);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(tao->solution, mad->isXU, &Wtmp);CHKERRQ(ierr);
        mad->Q->nF += 2;  mad->Q->nP += 1;  mad->Q->nS +=1;  mad->Q->nY += 1;
      }
      /* IPM can only work with filter so disable line search */
      mad->use_filter = PETSC_TRUE;
      ierr = TaoLineSearchDestroy(&tao->linesearch);CHKERRQ(ierr);
    } else {
      /* we're doing active-set, so let's create some placeholder IS objects */
      ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->isXL);CHKERRQ(ierr);
      ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->isXU);CHKERRQ(ierr);
      ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->fixedXB);CHKERRQ(ierr);
      ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->activeXB);CHKERRQ(ierr);
      ierr = ISCreate(PetscObjectComm((PetscObject)tao), &mad->inactiveXB);CHKERRQ(ierr);
    }
  }
  /* compute sizing of combined vectors */
  mad->Ns = mad->Ni + mad->Ncl + mad->Ncu + mad->Nxl + mad->Nxu;
  mad->Np = mad->Nx + mad->Ns;
  mad->Ny = mad->Ne + mad->Ni + mad->Ncl + mad->Ncu + mad->Nxl + mad->Nxu;
  mad->Nf = mad->Np + mad->Ny;
  mad->Nr = mad->Nx + mad->Ne + mad->Ni;

  /* at this point we should have created all the base vectors for Q */
  /* now we need to construct the VECNEST combinations */
  ierr = FullSpaceVecCreate(mad->Q);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->Qprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->Qwork);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->Qtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->D);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQ);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQwork);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->dLdQtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDuplicate(mad->Q, mad->W);CHKERRQ(ierr);

  /* create reduced space counterpart */
  ierr = ReducedSpaceVecCreate(mad->G);CHKERRQ(ierr);
  if (!mad->use_filter) {
    ierr = TaoLineSearchSetObjectiveAndGradientRoutine(tao->linesearch, TaoMADLineSearchObjAndGrad, (void*)tao);CHKERRQ(ierr);
    if (tao->bounded || tao->ineq_constrained) {
      ierr = PetscNewLog(tao, &mad->LB);CHKERRQ(ierr);
      ierr = ReducedSpaceVecDuplicate(mad->G, mad->LB);CHKERRQ(ierr);
      ierr = PetscNewLog(tao, &mad->UB);CHKERRQ(ierr);
      ierr = ReducedSpaceVecDuplicate(mad->G, mad->UB);CHKERRQ(ierr);
      ierr = VecCopy(tao->XL, mad->LB->X);CHKERRQ(ierr);
      ierr = VecCopy(tao->XU, mad->UB->X);CHKERRQ(ierr);
      if (tao->ineq_constrained) {
        ierr = VecSet(tao->IL, 0.0);CHKERRQ(ierr);
        ierr = VecCopy(tao->IL, mad->LB->Yi);CHKERRQ(ierr);
        ierr = VecSet(tao->IU, PETSC_INFINITY);CHKERRQ(ierr);
        ierr = VecCopy(tao->IU, mad->UB->Yi);CHKERRQ(ierr);
      }
      if (tao->eq_constrained) {
        ierr = VecSet(mad->LB->Ye, PETSC_NINFINITY);CHKERRQ(ierr);
        ierr = VecSet(mad->UB->Ye, PETSC_INFINITY);CHKERRQ(ierr);
      }
      ierr = TaoLineSearchSetVariableBounds(tao->linesearch, mad->LB->R, mad->UB->R);CHKERRQ(ierr);
    }
  }

  /* duplicate off vectors exclusive to SNES */
  ierr = VecDuplicate(mad->Q->R, &mad->Rsnes);CHKERRQ(ierr);
  ierr = VecDuplicate(mad->G->R, &mad->Gsnes);CHKERRQ(ierr);

  /* use offload the constructed reduced KKT problem to SNESNGMRES */
  ierr = SNESGetDM(mad->snes, &dm);CHKERRQ(ierr);
  ierr = DMShellSetGlobalVector(dm, mad->Rsnes);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(mad->snes, (void*)tao);CHKERRQ(ierr);
  ierr = SNESSetFunction(mad->snes, mad->Gsnes, TaoMADSNESFunction, (void*)tao);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(mad->snes, TaoMADSNESCheck, (void*) tao, NULL);CHKERRQ(ierr);
  ierr = SNESSetTolerances(mad->snes, tao->gatol, tao->grtol, tao->steptol, tao->max_it, tao->max_funcs);CHKERRQ(ierr);
  ierr = SNESSetUp(mad->snes);CHKERRQ(ierr);
  ierr = SNESGetLineSearch(mad->snes, &linesearch);CHKERRQ(ierr);
  if (mad->use_filter) {
    ierr = TaoLineSearchDestroy(&tao->linesearch);CHKERRQ(ierr);
    if (mad->filter_type == TAO_MAD_FILTER_SNES) {
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHCP);CHKERRQ(ierr);
      ierr = SNESLineSearchSetPreCheck(linesearch, TaoMADSNESLineSearchPreCheck, (void*)tao);CHKERRQ(ierr);
      ierr = SNESLineSearchSetPostCheck(linesearch, TaoMADSNESLineSearchPostCheck, (void*)tao);CHKERRQ(ierr);
    } else {
      ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL);CHKERRQ(ierr);
      ierr = SNESLineSearchShellSetUserFunc(linesearch, TaoMADSNESLineSearchApply, (void*)tao);CHKERRQ(ierr);
    }
  } else {
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHSHELL);CHKERRQ(ierr);
    ierr = SNESLineSearchShellSetUserFunc(linesearch, TaoMADSNESLineSearchApply, (void*)tao);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_MAD(Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  SimpleFilter   *filter = mad->filter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* destroy vector structures */
  ierr = FullSpaceVecDestroy(mad->Q);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->Qprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->Qwork);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->Qtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->D);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQ);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQprev);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQwork);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->dLdQtrial);CHKERRQ(ierr);
  ierr = FullSpaceVecDestroy(mad->W);CHKERRQ(ierr);
  ierr = ReducedSpaceVecDestroy(mad->G);CHKERRQ(ierr);
  /* destroy index sets and intermediate vectors */
  ierr = VecDestroy(&mad->CeScale);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->CiScale);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->isXL);CHKERRQ(ierr);
  ierr = ISDestroy(&mad->isXU);CHKERRQ(ierr);
  if (mad->use_ipm) {
    ierr = ISDestroy(&mad->isIL);CHKERRQ(ierr);
    ierr = ISDestroy(&mad->isIU);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->XL);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->XU);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->IL);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->IU);CHKERRQ(ierr);
    ierr = VecDestroy(&mad->B);CHKERRQ(ierr);
  } else {
    ierr = ISDestroy(&mad->fixedXB);CHKERRQ(ierr);
    ierr = ISDestroy(&mad->activeXB);CHKERRQ(ierr);
    ierr = ISDestroy(&mad->inactiveXB);CHKERRQ(ierr);
    ierr = ISDestroy(&mad->inactiveCI);CHKERRQ(ierr);
  }
  /* destroy filter */
  ierr = PetscFree2(filter->f, filter->h);CHKERRQ(ierr);
  ierr = PetscFree(filter);CHKERRQ(ierr);
  /* destroy Lagrangians */
  ierr = PetscFree(mad->L);CHKERRQ(ierr);
  ierr = PetscFree(mad->Lprev);CHKERRQ(ierr);
  ierr = PetscFree(mad->Lwork);CHKERRQ(ierr);
  ierr = PetscFree(mad->Ltrial);CHKERRQ(ierr);
  /* destroy SNES data */
  ierr = VecDestroy(&mad->Rsnes);CHKERRQ(ierr);
  ierr = VecDestroy(&mad->Gsnes);CHKERRQ(ierr);
  ierr = SNESDestroy(&mad->snes);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_MAD(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_MAD        *mad = (TAO_MAD*)tao->data;
  PetscBool      extra_monitor = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Multisecant Accelerated Descent for solving noisy nonlinear optimization problems with general constraints.");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_max_scale","maximum scaling factor for primal gradient and dual variables","",mad->scale_max,&mad->scale_max,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_mad_use_ipm","uses an interior-point formulation for the KKT conditions (disables linesearch)","",mad->use_ipm,&mad->use_ipm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_mad_use_filter","disables the linesearch and uses a filter instead","",mad->use_ipm,&mad->use_ipm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_mad_filter_type","globalize the solution with a simple filter","TaoMADFilterType",TaoMADFilters,(PetscEnum)mad->filter_type,(PetscEnum*)&mad->filter_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_step_min","minimum step length for the filter globalization","",mad->alpha_min,&mad->alpha_min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_mad_step_factor","backtracking factor for the step length","",mad->alpha_fac,&mad->alpha_fac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_mad_monitor","enable special monitor for MAD penalty and FtB terms","",extra_monitor,&extra_monitor,NULL);CHKERRQ(ierr);
  if (!mad->use_filter && !mad->use_ipm) {
    ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(mad->snes);CHKERRQ(ierr);
  if (tao->numbermonitors && extra_monitor) {
    ierr = TaoSetMonitor(tao, TaoMADMonitor, (void*)PETSC_VIEWER_STDOUT_WORLD, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */

/*MC
  TaoMAD - Multisecant Accelerated Descent method for solving nonlinear optimization problems with general constraints.

  Options Database Keys:
+ -tao_mad_max_scale <real>                   - maximum scaling factor for primal gradient and dual variables (default: 100.0)
. -tao_mad_barrier_scale <real>               - affine scaling/centering factor for log-barrier parameter updates (default: 0.1)
. -tao_mad_barrier_steplength <real>          - steplength for the log-barrier parameter updates (default: 0.95)
. -tao_mad_use_ipm                            - uses an interior-point formulation for the KKT conditions (disables linesearch)
. -tao_mad_use_filter                         - replaces the line search with a simple Fletcher-Leyffer filter
. -tao_mad_filter_type <none,barrier,markov>  - filter type for globalization (default: none)
. -tao_mad_max_filter_size <int>              - maximum number of iterates stored in the filter (default: 300)
. -tao_mad_step_min <real>                    - minimum step length for filter globalization (default: 1e-8)
. -tao_mad_step_shrink <real>                 - backtracking interval for the step length (default: 0.1)
. -tao_mad_step_factor <real>                 - backtracking factor for the step length (default: 0.5)
- -tao_mad_armijo_epsilon <real>              - Armijo condition factor for backtracking (default: 1e-6)

  Level: advanced

  Notes:
  MAD is an experimental optimization method that uses a "multisecant" approximation of the KKT matrix for
  solving generally constrained optimization problems that exhibit noise/errors in the gradient.

  This implementation supports general inequality constraints with an interior-point formulation where the slack
  variables are reduced out of the problem with closed-form updates. The multisecant approximation is constructed
  only for the optimization variables and Lagrange multipliers. It also supports two-sided inequality constraints
  defined with TaoSetInequalityBounds(). When no inequality bounds are set, the algorithm assumes that the constraint
  is c_i(x) <= 0.

  .vb
  while unconverged
    update Q with q_k - q_{k-1}
    update G with g_k - g_{k-1} + beta(q_k - q_{k-1})
    solve gamma = argmin || g_k - G gamma||
    construct search direction d = -eta g - (Q - eta G)gamma
    globalize primal variables with a simple filter
    update multipliers with fraction-to-the-boundary rule
  endwhile
  .ve

.seealso:
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_MAD(Tao tao)
{
  TAO_MAD          *mad;
  SimpleFilter     *filter;
  Lagrangian       *L, *Lprev, *Lwork, *Ltrial;
  ReducedSpaceVec  *G;
  FullSpaceVec     *Q, *Qprev, *Qwork, *Qtrial, *D;
  FullSpaceVec     *dLdQ, *dLdQprev, *dLdQwork, *dLdQtrial, *W;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao, &mad);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &G);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Q);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Qprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Qwork);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Qtrial);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &D);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQ);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQwork);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &dLdQtrial);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &W);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &filter);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &L);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Lprev);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Lwork);CHKERRQ(ierr);
  ierr = PetscNewLog(tao, &Ltrial);CHKERRQ(ierr);

  tao->ops->destroy        = TaoDestroy_MAD;
  tao->ops->setup          = TaoSetUp_MAD;
  tao->ops->setfromoptions = TaoSetFromOptions_MAD;
  tao->ops->view           = TaoView_MAD;
  tao->ops->solve          = TaoSolve_MAD;

  tao->gatol = 1.e-5;
  tao->grtol = 0.0;
  tao->gttol = 0.0;
  tao->catol = 1.e-5;
  tao->crtol = 0.0;

  tao->data           = (void*)mad;
  mad->mu             = 100.0;
  mad->mu_r           = 0.95;
  mad->mu_g           = 0.1;
  mad->mu_min         = PETSC_MACHINE_EPSILON;
  mad->mu_max         = 1e5;
  mad->scale_max      = 100.0;
  mad->slack_init     = 1.0;
  mad->alpha_min      = 1e-5;
  mad->alpha_fac      = 0.5;
  mad->tau_min        = 0.99;
  mad->suff_decr      = 0.0;
  mad->unconstrained  = PETSC_TRUE;
  mad->use_ipm        = PETSC_FALSE;
  mad->use_filter     = PETSC_FALSE;
  mad->as_step        = 1.e-3;

  mad->Q              = Q;
  mad->Qprev          = Qprev;
  mad->Qwork          = Qwork;
  mad->Qtrial         = Qtrial;
  mad->D              = D;
  mad->dLdQ           = dLdQ;
  mad->dLdQprev       = dLdQprev;
  mad->dLdQwork       = dLdQwork;
  mad->dLdQtrial      = dLdQtrial;
  mad->W              = W;
  mad->G              = G;
  mad->L              = L;
  mad->Lprev          = Lprev;
  mad->Lwork          = Lwork;
  mad->Ltrial         = Ltrial;

  mad->filter_type    = TAO_MAD_FILTER_SNES;
  mad->filter         = filter;
  filter->max_size    = 100;
  filter->size        = 0;

  /*  set linear solver to default for symmetric matrices */
  ierr = KSPCreate(((PetscObject)tao)->comm,&tao->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->ksp, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(tao->ksp, tao->hdr.prefix);CHKERRQ(ierr);
  ierr = KSPSetType(tao->ksp, KSPCG);

  /* create the line search */
  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm,&tao->linesearch);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, TAOLINESEARCHMT);CHKERRQ(ierr);

  /*  create the SNES Anderson solver */
  ierr = SNESCreate(((PetscObject)tao)->comm,&mad->snes);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)mad->snes, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)mad->snes,"TAO");CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(mad->snes,"tao_mad_");CHKERRQ(ierr);
  ierr = SNESSetType(mad->snes, SNESNGMRES);CHKERRQ(ierr);
  ierr = SNESNGMRESSetRestartType(mad->snes, SNES_NGMRES_RESTART_DIFFERENCE);CHKERRQ(ierr);
  ierr = SNESNGMRESSetRestartFmRise(mad->snes, PETSC_TRUE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode TaoMADGetSNES(Tao tao, SNES* snes)
{
  TAO_MAD* mad = (TAO_MAD*)tao->data;

  PetscFunctionBegin;
  *snes = mad->snes;
  PetscFunctionReturn(0);
}