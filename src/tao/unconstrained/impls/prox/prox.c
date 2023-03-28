#include <petsctaolinesearch.h>
#include <../src/tao/unconstrained/impls/cg/prox.h>

static PetscErrorCode TaoSolve_PROX(Tao tao)
{
  TAO_PROX                     *proxP       = (TAO_PROX *)tao->data;
  TaoLineSearchConvergedReason  ls_status   = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                     step        = 1.0, f, gnorm, gnorm2, delta, gd, ginner, beta;
  PetscReal                     gd_old, gnorm2_old, f_old;

  PetscFunctionBegin;
  if (tao->XL || tao->XU || tao->ops->computebounds) PetscCall(PetscInfo(tao, "WARNING: Variable bounds have been set but will be ignored by prox algorithm\n"));

  /*  Check convergence criteria */
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /*  Set initial direction to -gradient */
  PetscCall(VecCopy(tao->gradient, tao->stepdirection));
  PetscCall(VecScale(tao->stepdirection, -1.0));
  gnorm2 = gnorm * gnorm;

  /*  Set initial scaling for the function */
  if (f != 0.0) {
    delta = 2.0 * PetscAbsScalar(f) / gnorm2;
    delta = PetscMax(delta, proxP->delta_min);
    delta = PetscMin(delta, proxP->delta_max);
  } else {
    delta = 2.0 / gnorm2;
    delta = PetscMax(delta, proxP->delta_min);
    delta = PetscMin(delta, proxP->delta_max);
  }
  /*  Set counter for gradient and reset steps */
  proxP->ngradsteps  = 0;
  proxP->nresetsteps = 0;

  while (1) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);

    /*  Save the current gradient information */
    f_old      = f;
    gnorm2_old = gnorm2;
    PetscCall(VecCopy(tao->solution, proxP->X_old));
    PetscCall(VecCopy(tao->gradient, proxP->G_old));
    PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
    if ((gd >= 0) || PetscIsInfOrNanReal(gd)) {
      ++proxP->ngradsteps;
      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta, proxP->delta_min);
        delta = PetscMin(delta, proxP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta, proxP->delta_min);
        delta = PetscMin(delta, proxP->delta_max);
      }

      PetscCall(VecCopy(tao->gradient, tao->stepdirection));
      PetscCall(VecScale(tao->stepdirection, -1.0));
    }

    /*  Search direction for improving point */
    PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
    PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
    PetscCall(TaoAddLineSearchCounts(tao));
    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /*  Linesearch failed */
      /*  Reset factors and use scaled gradient step */
      ++proxP->nresetsteps;
      f      = f_old;
      gnorm2 = gnorm2_old;
      PetscCall(VecCopy(proxP->X_old, tao->solution));
      PetscCall(VecCopy(proxP->G_old, tao->gradient));

      if (f != 0.0) {
        delta = 2.0 * PetscAbsScalar(f) / gnorm2;
        delta = PetscMax(delta, proxP->delta_min);
        delta = PetscMin(delta, proxP->delta_max);
      } else {
        delta = 2.0 / gnorm2;
        delta = PetscMax(delta, proxP->delta_min);
        delta = PetscMin(delta, proxP->delta_max);
      }

      PetscCall(VecCopy(tao->gradient, tao->stepdirection));
      PetscCall(VecScale(tao->stepdirection, -1.0));

      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
      PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        /*  Linesearch failed again */
        /*  switch to unscaled gradient */
        f = f_old;
        PetscCall(VecCopy(proxP->X_old, tao->solution));
        PetscCall(VecCopy(proxP->G_old, tao->gradient));
        delta = 1.0;
        PetscCall(VecCopy(tao->solution, tao->stepdirection));
        PetscCall(VecScale(tao->stepdirection, -1.0));

        PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, delta));
        PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, tao->stepdirection, &step, &ls_status));
        PetscCall(TaoAddLineSearchCounts(tao));
        if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
          /*  Line search failed for last time -- give up */
          f = f_old;
          PetscCall(VecCopy(proxP->X_old, tao->solution));
          PetscCall(VecCopy(proxP->G_old, tao->gradient));
          step        = 0.0;
          tao->reason = TAO_DIVERGED_LS_FAILURE;
        }
      }
    }

    /*  Check for bad value */
    PetscCall(VecNorm(tao->gradient, NORM_2, &gnorm));
    PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User-provided compute function generated Inf or NaN");

    /*  Check for termination */
    gnorm2 = gnorm * gnorm;
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    /*  Check for restart condition */
    PetscCall(VecDot(tao->gradient, proxP->G_old, &ginner));
    if (PetscAbsScalar(ginner) >= proxP->eta * gnorm2) {
      /*  Gradients far from orthogonal; use steepest descent direction */
      beta = 0.0;
    } else {
      /*  Gradients close to orthogonal; use conjugate gradient formula */
      switch (proxP->cg_type) {
      case PROX_FletcherReeves:
        beta = gnorm2 / gnorm2_old;
        break;

      case PROX_PolakRibiere:
        beta = (gnorm2 - ginner) / gnorm2_old;
        break;

      case PROX_PolakRibierePlus:
        beta = PetscMax((gnorm2 - ginner) / gnorm2_old, 0.0);
        break;

      case PROX_HestenesStiefel:
        PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
        PetscCall(VecDot(proxP->G_old, tao->stepdirection, &gd_old));
        beta = (gnorm2 - ginner) / (gd - gd_old);
        break;

      case PROX_DaiYuan:
        PetscCall(VecDot(tao->gradient, tao->stepdirection, &gd));
        PetscCall(VecDot(proxP->G_old, tao->stepdirection, &gd_old));
        beta = gnorm2 / (gd - gd_old);
        break;

      default:
        beta = 0.0;
        break;
      }
    }

    /*  Compute the direction d=-g + beta*d */
    PetscCall(VecAXPBY(tao->stepdirection, -1.0, beta, tao->gradient));

    /*  update initial steplength choice */
    delta = 1.0;
    delta = PetscMax(delta, proxP->delta_min);
    delta = PetscMin(delta, proxP->delta_max);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!tao->stepdirection) PetscCall(VecDuplicate(tao->solution, &tao->stepdirection));
  if (!proxP->X_old) PetscCall(VecDuplicate(tao->solution, &proxP->X_old));
  if (!proxP->G_old) PetscCall(VecDuplicate(tao->gradient, &proxP->G_old));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_PROX(Tao tao)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroy(&proxP->X_old));
    PetscCall(VecDestroy(&proxP->G_old));
  }
  PetscCall(TaoLineSearchDestroy(&tao->linesearch));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_PROX(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_PROX *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadBegin(PetscOptionsObject, "Nonlinear Conjugate Gradient method for unconstrained optimization");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_PROX(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_PROX   *proxP = (TAO_PROX *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PROX Type: %s\n", PROX_Table[proxP->prox_type]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Gradient steps: %" PetscInt_FMT "\n", proxP->ngradsteps));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Reset steps: %" PetscInt_FMT "\n", proxP->nresetsteps));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     TAOPROX -  Proximal algorithm.

   Options Database Keys:

  Notes:
     PROX formulas are:
  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_PROX(Tao tao)
{
  TAO_PROX     *proxP;
  const char *morethuente_type = TAOLINESEARCHMT;

  PetscFunctionBegin;
  tao->ops->setup          = TaoSetUp_PROX;
  tao->ops->solve          = TaoSolve_PROX;
  tao->ops->view           = TaoView_PROX;
  tao->ops->setfromoptions = TaoSetFromOptions_PROX;
  tao->ops->destroy        = TaoDestroy_PROX;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  /*  Note: nondefault values should be used for nonlinear conjugate gradient  */
  /*  method.  In particular, gtol should be less that 0.5; the value used in  */
  /*  Nocedal and Wright is 0.10.  We use the default values for the  */
  /*  linesearch because it seems to work better. */
  PetscCall(TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, morethuente_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));

  PetscCall(PetscNew(&proxP));
  tao->data      = (void *)proxP;
  proxP->eta       = 0.1;
  proxP->delta_min = 1e-7;
  proxP->delta_max = 100;
  PetscFunctionReturn(PETSC_SUCCESS);
}
