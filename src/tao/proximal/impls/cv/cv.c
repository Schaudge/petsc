#include <../src/tao/proximal/impls/cv/cv.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] = "@article{latafat2023adaptive,\n"
                               "title={Adaptive proximal algorithms for convex optimization under local Lipschitz continuity of the gradient},\n"
                               "author={Latafat, Puya and Themelis, Andreas and Stella, Lorenzo and Patrinos, Panagiotis},\n"
                               "journal={arXiv preprint arXiv:2301.04431},\n"
                               "pages={4},\n"
                               "year={2023}"
                               "}\n";
#if 0
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
#endif

static PetscErrorCode TaoCV_ObjGrad_Private(Tao tao, Vec x, PetscReal *f, Vec g)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  if (cv->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(cv->smoothterm, x, f, g));
    *f *= cv->f_scale;
    PetscCall(VecScale(g, cv->f_scale));
  } else {
    PetscCall(TaoComputeObjectiveAndGradient(tao, x, f, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* if operator norm of linear map is not set, we estimate it, and store the
 * estimate inside of cv->h_lmap_norm. */
//TODO I have this implemented in PSARMIJO too, but...
static PetscErrorCode TaoCV_Stepsize_With_LS_Private(Tao tao)
{
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscReal xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3, temp, temp2, temp3, step_new, rho, norm1, norm2;

  PetscFunctionBegin;
  // TODO should we use eta for norm estimate, or just use h_map_norm? set bool is still false
  xi  = cv->pd_ratio * tao->step * cv->eta * (1+tao->gatol);
  xi *= xi;

  PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, tao->gradient));
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, tao->solution));
  PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
  PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

  L = (xdiffnorm == 0) ? 0 : grad_x_dot / (xdiffnorm*xdiffnorm);
  C = (grad_x_dot == 0) ? 0 : (graddiffnorm*graddiffnorm) / grad_x_dot;
  D = tao->step* L * (tao->step * C - 1);

//  cv->h_lmap_norm *= cv->R;
  cv->eta *= cv->R;

  PetscInt iter = 0;
  while (1) {
    min1      = tao->step * PetscSqrtReal(1 + tao->step / cv->step_old);
    min2      = 1 / (2 * cv->nu * cv->pd_ratio * cv->eta);
    temp      = 1 - 4*xi;
    temp2     = cv->pd_ratio * cv->eta * tao->step; //Unlike no linesearch, this "xi" uses updated norm estimate
    temp3     = PetscSqrtReal(D*D + temp*temp2*temp2);
    min3      = tao->step * PetscSqrtReal(temp / (2*(1+tao->gatol)*(temp3 +D)));
    step_new  = PetscMin(min1, PetscMin(min2, min3));
    rho       = step_new / tao->step;
    cv->sigma = cv->pd_ratio*cv->pd_ratio*step_new;

    /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
    PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, cv->dualvec));
    PetscCall(VecAXPY(cv->dualvec_work, cv->sigma*(1+rho), cv->Ax));
    /* dualvec: y = prox_h*(w, sigma) */
    PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, cv->sigma, cv->dualvec_work, cv->dualvec_test, PETSC_TRUE));
    /* workvec : A^T * y_test */
    PetscCall(MatMultTranspose(cv->h_lmap, cv->dualvec_test, cv->workvec));
    /* norm1 = norm(ATy_test - ATy) */
    PetscCall(VecWAXPY(cv->workvec2, -1., cv->ATy, cv->workvec));
    PetscCall(VecNorm(cv->workvec2, NORM_2, &norm1));
    /* norm2 = norm(y_test - y) */
    PetscCall(VecWAXPY(cv->dualvec_work2, -1., cv->dualvec_test, cv->dualvec));
    PetscCall(VecNorm(cv->dualvec_work2, NORM_2, &norm2));
    if (cv->eta >= norm1 / norm2) {
      cv->step_old = tao->step;
      tao->step    = step_new;
      PetscCall(VecCopy(cv->dualvec_test, cv->dualvec));
      PetscCall(VecCopy(cv->workvec, cv->ATy));
      break;
    }
    cv->eta *= cv->r;
    iter++;
  }
  printf(" end of ls, iter %d \n", iter);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoCV_Stepsize_No_LS_Private(Tao tao)
{
  TAO_CV   *cv = (TAO_CV *)tao->data;
  PetscReal xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3, temp, temp2;

  PetscFunctionBegin;
  xi  = cv->pd_ratio * tao->step * cv->h_lmap_norm;
  xi *= xi;

  PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, tao->gradient));
  PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, tao->solution));
  PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
  PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
  PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

  L = grad_x_dot / (xdiffnorm*xdiffnorm);
  C = (graddiffnorm*graddiffnorm) / grad_x_dot;
  D = tao->step* L * (tao->step * C - 1);

  min1         = tao->step * PetscSqrtReal(1 + tao->step / cv->step_old);
  min2         = 1 / (2 * cv->nu * cv->pd_ratio * cv->h_lmap_norm);
  temp         = 1 - 4*xi*(1+tao->gatol)*(1+tao->gatol);
  temp2        = PetscSqrtReal(D*D + xi*temp);
  min3         = tao->step * PetscSqrtReal(temp / (2*(1+tao->gatol)*(temp2 +D)));
  cv->step_old = tao->step;
  tao->step    = PetscMin(min1, PetscMin(min2, min3));
  cv->sigma    = cv->pd_ratio*cv->pd_ratio*tao->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_CV(Tao tao)
{
  TAO_CV    *cv = (TAO_CV *)tao->data;
  PetscReal  f, gnorm, lip, rho;
  PetscReal  pri_res_norm, dual_res_norm;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(cv->R <= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Scale factor needs to be equal or less than 1");
  PetscCheck(cv->r > 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracking factor needs to be greater than 1");
  PetscCheck(!(cv->use_accel && cv->use_adapt), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoCV only supports either acceleration or adaptive step, not both");
  if (cv->smoothterm) PetscCall(DMTaoGetLipschitz(cv->smoothterm, &lip));
  else lip = 0.;
  PetscCall(PetscCitationsRegister(citation, &cited));

  cv->lip = lip;

  if (tao->step == 0 && cv->h_lmap_norm > 0) tao->step = 1 / (2* cv->nu * cv->pd_ratio * cv->h_lmap_norm);
  else if (tao->step == 0 && cv->h_lmap_norm == 0) tao->step = 1 / (2* cv->nu * cv->pd_ratio * cv->eta);
  cv->sigma = tao->step* cv->pd_ratio * cv->pd_ratio;

  cv->step_old = tao->step;

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));
  PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  /* set initial dualvec, y as zero, so skip initial ATy computation */
  tao->reason = TAO_CONTINUE_ITERATING;
  PetscCall(VecCopy(tao->solution, cv->x_old));
  PetscCall(VecCopy(tao->gradient, cv->grad_old));
  PetscCall(VecCopy(cv->Ax, cv->Ax_old));
  PetscCall(VecSet(cv->ATy, 0.));
  PetscCall(VecSet(cv->dualvec, 0.));

  PetscCall(MatMult(cv->h_lmap, tao->solution, cv->Ax));
  PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* workvec: v = x - step * (grad_x + ATy) */
    PetscCall(VecWAXPY(cv->workvec, -tao->step, tao->gradient, tao->solution));
    PetscCall(VecAXPY(cv->workvec, -tao->step, cv->ATy));

    /* x = prog_g(v, step) */
    PetscCall(VecCopy(tao->solution, cv->x_old));//TODO Axprev, gradprev copy here
    PetscCall(DMTaoApplyProximalMap(cv->g_prox, cv->reg, tao->step*cv->g_scale, cv->workvec, tao->solution, PETSC_FALSE));
    /* update Ax, and grad */
    PetscCall(VecCopy(cv->Ax, cv->Ax_old));
    PetscCall(VecCopy(tao->gradient, cv->grad_old));
    PetscCall(MatMult(cv->h_lmap, tao->solution, cv->Ax));
    PetscCall(TaoCV_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));
    /* workvec = (v - x)/step + grad_x + ATy */
    PetscCall(VecAXPY(cv->workvec, -1., tao->solution));
    PetscCall(VecScale(cv->workvec, 1/tao->step));
    PetscCall(VecAXPBYPCZ(cv->workvec, 1., 1., 1., tao->gradient, cv->ATy));
    PetscCall(VecNorm(cv->workvec, NORM_2, &pri_res_norm));

    // update stepsize
    //TODO should we bother with non-adaptive CV?
    if (cv->lmap_norm_set) {
      PetscCall(TaoCV_Stepsize_No_LS_Private(tao));
      rho = tao->step / cv->step_old;

      /* dualvec_work: w = y + sigma *((1+rho) * Ax - rho * Ax_old) */
      PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma*rho, cv->Ax_old, cv->dualvec));
      PetscCall(VecAXPY(cv->dualvec_work, cv->sigma * (1+rho), cv->Ax));

      /* dualvec: y = prox_h*(w, sigma) */
      PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, cv->sigma*cv->h_scale, cv->dualvec_work, cv->dualvec, PETSC_TRUE));
    } else {
      //PetscCall(TaoLineSearchApply(tao->linesearch, tao->solution, &f, tao->gradient, fb->dualvec, &tao->step, &ls_status));
      //calling this a linesearch a-la Armijo is a strech....
      PetscCall(TaoCV_Stepsize_With_LS_Private(tao)); //prox_h is done inside this routine for linesearch version
    }

    PetscCall(VecAXPY(cv->dualvec_work, -1., cv->dualvec));
    PetscCall(VecScale(cv->dualvec_work, 1/cv->sigma));
    PetscCall(VecAXPY(cv->dualvec_work, -1., cv->Ax));
    PetscCall(VecNorm(cv->dualvec_work, NORM_2, &dual_res_norm));

    tao->residual = PetscSqrtReal(pri_res_norm*pri_res_norm) + PetscSqrtReal(dual_res_norm*dual_res_norm);

    /* convergence test */
    PetscCall(TaoLogConvergenceHistory(tao, f, tao->residual, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, tao->residual, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    tao->niter++;

    /* post-processing */
    PetscCall(MatMultTranspose(cv->h_lmap, cv->dualvec, cv->ATy));//TODO dont need this for LS version.
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_CV(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(Ax)+h(x), where you have gradient of f(x), and proximal operator of g(x) and h(x), and some linear map A.");
  PetscCall(PetscOptionsReal("-tao_cv_initial_step", "Initial stepsize for Condat-Vu algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_norm_estimate_factor", "Scale factor for estimating operator norm of linear map. Must be <= 1.", "", cv->R, &cv->R, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_primal_dual_ratio", "Primal-dual Ratio factor for balancing solution. Must be non-negative", "", cv->pd_ratio, &cv->pd_ratio, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_backtrack_parameter", "Backtracking parameter r. Must be  >1.", "", cv->r, &cv->r, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_nu", "Stepsize scale parameter nu. Must be  >1+tol.", "", cv->nu, &cv->nu, NULL));
  //TODO TaoCVSetInitialNormEstimate(Tao, PetscReal) ?
  PetscCall(PetscOptionsReal("-tao_cv_eta", "Initial linear map norm estimate. Must be nonnegative", "", cv->eta, &cv->eta, NULL));
  PetscCall(PetscOptionsReal("-tao_cv_tol", "Stepsize tolerance parameter", "", cv->tol, &cv->tol, NULL));
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
    if (cv->g_prox) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Non-smooth Term g(x):\n"));
      PetscCall(DMTaoView(cv->g_prox, viewer));
    }
    if (cv->g_prox) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Non-smooth Term g(Ax):\n"));
      PetscCall(DMTaoView(cv->g_prox, viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Linear map of Non-smooth Term h(Ax):\n"));
      PetscCall(MatView(cv->h_lmap, viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_CV(Tao tao)
{
  TAO_CV *cv = (TAO_CV *)tao->data;

  PetscFunctionBegin;
  /* sol sized vectors */
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!cv->workvec) PetscCall(VecDuplicate(tao->solution, &cv->workvec));
  if (!cv->workvec2) PetscCall(VecDuplicate(tao->solution, &cv->workvec2));
  if (!cv->x_old) PetscCall(VecDuplicate(tao->solution, &cv->x_old));
  if (!cv->grad_old) PetscCall(VecDuplicate(tao->solution, &cv->grad_old));
  if (!cv->ATy) PetscCall(VecDuplicate(tao->solution, &cv->ATy));
  if (!cv->ATy_test || !cv->lmap_norm_set) PetscCall(VecDuplicate(tao->solution, &cv->ATy_test));//TODO ?
  /* dual sized vectors */
  if (!cv->Ax) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->Ax));
  if (!cv->Ax_old) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->Ax_old));
  if (!cv->dualvec) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec));
  if (!cv->dualvec_test || !cv->lmap_norm_set) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_test));
  if (!cv->dualvec_work) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_work));
  if (!cv->dualvec_work2) PetscCall(MatCreateVecs(cv->h_lmap, NULL, &cv->dualvec_work2));
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
  PetscCall(VecDestroy(&cv->Ax));
  PetscCall(VecDestroy(&cv->Ax_old));
  PetscCall(VecDestroy(&cv->ATy));
  PetscCall(VecDestroy(&cv->ATy_test));
  PetscCall(VecDestroy(&cv->dualvec));
  PetscCall(VecDestroy(&cv->dualvec_work));
  PetscCall(VecDestroy(&cv->dualvec_work2));
  PetscCall(VecDestroy(&cv->dualvec_test));
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
//  tao->ops->convergencetest   = TaoConvergenceTest_CV;
  tao->ops->convergencetest   = TaoDefaultConvergenceTest;

  tao->data = (void *)cv;

  cv->gnorm_norm  = 0.;
  cv->tol         = 1.e-6;
  cv->R           = 0.95;
  cv->r           = 2.;
  cv->pd_ratio    = 0.01;
  cv->nu          = 1.2;
  cv->eta         = 1.;
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
