#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

static PetscErrorCode TaoConvergenceTest_CV(Tao tao, void *dummy)
{
  TAO_CV            *cv    = (TAO_CV *)tao->data;
  PetscInt           niter = tao->niter, nfuncs = PetscMax(tao->nfuncs, tao->nfuncgrads);
  PetscInt           max_funcs = tao->max_funcs;
  PetscReal          gnorm = tao->residual, gnorm0 = tao->gnorm0;
  PetscReal          f     = tao->fc;
  PetscReal          gatol = tao->gatol, grtol = tao->grtol, gttol = tao->gttol;
  PetscReal          catol = tao->catol, crtol = tao->crtol;
  PetscReal          fmin = tao->fmin, cnorm = tao->cnorm;
  TaoConvergedReason reason = tao->reason;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  if (PetscIsInfOrNanReal(f)) {
    PetscCall(PetscInfo(tao, "Failed to converged, function value is Inf or NaN\n"));
    reason = TAO_DIVERGED_NAN;
  } else if (f <= fmin && cnorm <= catol) {
    PetscCall(PetscInfo(tao, "Converged due to function value %g < minimum function value %g\n", (double)f, (double)fmin));
    reason = TAO_CONVERGED_MINF;
  } else if (gnorm <= gatol && cnorm <= catol) {
    PetscCall(PetscInfo(tao, "Converged due to residual norm ||g(X)||=%g < %g\n", (double)gnorm, (double)gatol));
    reason = TAO_CONVERGED_GATOL;
  } else if (f != 0 && PetscAbsReal(gnorm / cv->gnorm_norm) <= grtol && cnorm <= crtol) {
    PetscCall(PetscInfo(tao, "Converged due to residual ||g(X)||/|f(X)| =%g < %g\n", (double)(gnorm / cv->gnorm_norm), (double)grtol));
    reason = TAO_CONVERGED_GRTOL;
  } else if (gnorm0 != 0 && ((gttol == 0 && gnorm == 0) || gnorm / gnorm0 < gttol) && cnorm <= crtol) {
    PetscCall(PetscInfo(tao, "Converged due to relative residual norm ||g(X)||/||g(X0)|| = %g < %g\n", (double)(gnorm / gnorm0), (double)gttol));
    reason = TAO_CONVERGED_GTTOL;
  } else if (max_funcs >= 0 && nfuncs > max_funcs) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of function evaluations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", nfuncs, max_funcs));
    reason = TAO_DIVERGED_MAXFCN;
  } else if (tao->lsflag != 0) {
    PetscCall(PetscInfo(tao, "Tao Line Search failure.\n"));
    reason = TAO_DIVERGED_LS_FAILURE;
  } else if (niter >= tao->max_it) {
    PetscCall(PetscInfo(tao, "Exceeded maximum number of iterations: %" PetscInt_FMT " > %" PetscInt_FMT "\n", niter, tao->max_it));
    reason = TAO_DIVERGED_MAXITS;
  } else {
    reason = TAO_CONTINUE_ITERATING;
  }
  tao->reason = reason;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_CV(Tao tao)
{
  TAO_CV                      *cv = (TAO_CV *)tao->data;
  PetscReal                    f, gnorm, lip;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(!(cv->use_accel && cv->use_adapt), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoCV only supports either acceleration or adaptive step, not both");
  PetscCall(DMTaoGetLipschitz(cv->smoothterm, &lip));

  cv->lip = lip;

  if (cv->approx_lip || cv->lip == 0) {
    /* Approximating initial Lipschitz number via two random vectors */
    PetscReal   gradnorm, xnorm;
    PetscRandom rctx;

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(cv->workvec, rctx));
    PetscCall(VecSetRandom(cv->workvec2, rctx));//TODO are they different? check later
    /* If TaoCVSetSmoothTerm has been called, will use DM for gradient.
     * Otherwise, use one set via TaoSetGradient */
    if (cv->smoothterm) {
      PetscCall(DMTaoComputeGradient(cv->smoothterm, cv->workvec, cv->x_old));
      PetscCall(DMTaoComputeGradient(cv->smoothterm, cv->workvec2, cv->grad_old));
    } else {
      PetscCall(TaoComputeGradient(tao, cv->workvec, cv->x_old));
      PetscCall(TaoComputeGradient(tao, cv->workvec2, cv->grad_old));
    }
    PetscCall(VecAXPY(cv->grad_old, -1., cv->x_old));
    PetscCall(VecAXPY(cv->workvec, -1., cv->workvec2));
    PetscCall(VecNorm(cv->grad_old, NORM_2, &gradnorm));
    PetscCall(VecNorm(cv->workvec, NORM_2, &xnorm));

    cv->lip   = gradnorm / xnorm;
    cv->lip   = PetscMax(cv->lip, 1.e-6);
    tao->step = 2. / cv->lip / 10.;

    PetscCall(PetscRandomDestroy(&rctx));
  } else if (cv->lip > 0) {
    tao->step = 1. / cv->lip;//TODO what if lip is set but want to use some set init step? There is no TaoSetInitialStep, so maybe TaoPSSetIntialStep? or former suffices?
  }

  cv->step_old = tao->step;

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  if (cv->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(cv->smoothterm, tao->solution, &f, tao->gradient));
  } else {
    PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  }
  PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  tao->reason = TAO_CONTINUE_ITERATING;

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscCall(VecCopy(tao->solution, cv->x_old));
    PetscCall(VecCopy(tao->gradient, cv->grad_old));

    tao->niter++;

    //Logging
    //TODO gnorm?
    PetscCall(TaoLogConvergenceHistory(tao, f, 0, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, 0, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    //TODO VM
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_CV(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(x), where you have gradient of f(x), and proximal operator of g(x).");
  PetscCall(PetscOptionsReal("-tao_cv_initial_step", "Initial stepsize for forward-backward algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsBool("-tao_cv_approx_lip", "Approximate Lipschitz in the beginning", "", cv->approx_lip, &cv->approx_lip, NULL));
  PetscCall(PetscOptionsBool("-tao_cv_accel", "Use Acceleration (Nesterov-type)", "", cv->use_accel, &cv->use_accel, NULL));
  PetscCall(PetscOptionsBool("-tao_cv_adaptive", "Use adaptive stepsize (adaPDM)", "", cv->use_adapt, &cv->use_adapt, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_CV(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_CV   *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (cv->smoothterm) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Smooth Term:\n"));
    PetscCall(DMTaoView(cv->smoothterm, viewer));
    }
    //TODO prox, map views
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!cv->workvec) PetscCall(VecDuplicate(tao->solution, &cv->workvec));
  if (!cv->workvec2) PetscCall(VecDuplicate(tao->solution, &cv->workvec2));
  if (!cv->x_old) PetscCall(VecDuplicate(tao->solution, &cv->x_old));
  if (!cv->grad_old) PetscCall(VecDuplicate(tao->solution, &cv->grad_old));
  //TODO option to set regularizer type??
  //if L is set, should default stepsize be that? not for sparsa...
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &cv->reg));
  PetscCall(DMTaoSetType(cv->reg, DMTAOL2));
  if (cv->smoothterm) {
    PetscCall(TaoLineSearchUseDM(tao->linesearch, cv->smoothterm));
  } else {
    PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&cv->workvec));
  PetscCall(VecDestroy(&cv->workvec2));
  PetscCall(VecDestroy(&cv->x_old));
  PetscCall(VecDestroy(&cv->grad_old));
  PetscCall(DMDestroy(&cv->reg));
  PetscCall(DMDestroy(&cv->smoothterm));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_CV(Tao tao)
{
  TAO_CV     *cv;
  const char *armijo_type = TAOLINESEARCHPSARMIJO;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cv));

  tao->ops->destroy           = TaoDestroy_CV;
  tao->ops->setup             = TaoSetUp_CV;
  tao->ops->setfromoptions    = TaoSetFromOptions_CV;
  tao->ops->view              = TaoView_CV;
  tao->ops->solve             = TaoSolve_CV;
  tao->ops->convergencetest   = TaoConvergenceTest_CV;

  tao->data = (void *)cv;

  cv->gnorm_norm  = 0.;
  cv->smoothterm  = NULL;
  cv->approx_lip  = PETSC_TRUE;
  cv->use_accel   = PETSC_TRUE;
  cv->use_adapt   = PETSC_FALSE;

  /* Non-monotonic linesearch
   *
   * \hat{f}_k = max(f_{k-1},f_{k-2}, ... , f_{k-min(M,k)}), where M is linesearch history size
   * f(x_{k+1}) < \hat{f}_k + \langle x_{k+1} - x_k, grad_f(x_k) \rangle + (1/(2 step)) |x_{k+1} - x_k |^2
   *
   */
  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, armijo_type));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}
