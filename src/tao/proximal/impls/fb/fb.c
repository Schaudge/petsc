#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

static PetscBool  fasta_cited       = PETSC_FALSE;
static PetscBool  adapgm_cited      = PETSC_FALSE;
static const char fasta_citation[]  = "@article{goldstein2015fasta,\n"
                                      "title={FASTA: A generalized implementation of forward-backward splitting},\n"
                                      "author={Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},\n"
                                      "journal={arXiv preprint arXiv:1501.04979},\n"
                                      " year={2015}\n"
                                      "}\n";
static const char adapgm_citation[] = "@article{latafat2023convergence,\n"
                                      "title={On the convergence of adaptive first order methods: proximal gradient and alternating minimization algorithms},\n"
                                      "author={Latafat, Puya and Themelis, Andreas and Patrinos, Panagiotis},\n"
                                      "journal={arXiv preprint arXiv:2311.18431},\n"
                                      "year={2023}\n"
                                      "}\n";

#if 0
PetscErrorCode TaoFBSetUseLipApprox(Tao tao, PetscBool flag)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  fb->approx_lip = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBSetNonSmoothTerm(Tao tao, DM dm)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  fb->proxterm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBSetSmoothTerm(Tao tao, DM dm)
{
  TAO_FB *fb = (TAO_FB *)tao->data;
  DMTao   tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCheck(tdm->ops->computeobjectiveandgradient || tdm->ops->computegradient, PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "DMTaoSetGradient() has not been called");
  PetscCall(PetscObjectReference((PetscObject)dm));
  fb->smoothterm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoConvergenceTest_FB(Tao tao, void *dummy)
{
  TAO_FB            *fb    = (TAO_FB *)tao->data;
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
  } else if (f != 0 && PetscAbsReal(gnorm / fb->gnorm_norm) <= grtol && cnorm <= crtol) {
    PetscCall(PetscInfo(tao, "Converged due to residual ||g(X)||/|f(X)| =%g < %g\n", (double)(gnorm / fb->gnorm_norm), (double)grtol));
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

static PetscErrorCode TaoFB_ADAPGM_Update_Stepsize_Private(Tao tao)
{
  TAO_FB   *fb = (TAO_FB *)tao->data;
  PetscReal grad_x_dot, graddiff, xdiff, L, C, min1, min2;

  PetscFunctionBegin;
  /* workvec: v, which is x - step*gradf(x) */
  //temp \gets sqrt(norm(xnew - xold)
  /* Gradient eval happens before prox for adaPGM, but after prox for FISTA-variants */
  /* workvec: (z-x)/step + gradf(x) */

  // step update
  // workvec  : gradf(xnew) - gradf(xold)
  // workvec3 : xnew - xold
  PetscCall(VecWAXPY(fb->workvec, -1., fb->grad_old, tao->gradient));
  PetscCall(VecWAXPY(fb->workvec3, -1., fb->x_old, tao->solution));
  PetscCall(VecTDot(fb->workvec, fb->workvec3, &grad_x_dot));
  PetscCall(VecTDot(fb->workvec3, fb->workvec3, &xdiff));
  PetscCall(VecTDot(fb->workvec, fb->workvec, &graddiff));
  L = grad_x_dot / xdiff;
  C = graddiff / grad_x_dot;

  min1 = tao->step * PetscSqrtReal(1+ tao->step/fb->step_old);
  min2 = tao->step / (2 * PetscSqrtReal(PetscMax(tao->step * L * (tao->step * C - 1), 0)));

  fb->step_old = tao->step;
  tao->step    = PetscMin(min1, min2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_ComputeResidual_And_LogConv_Private(Tao tao, PetscReal f)
{
  TAO_FB    *fb = (TAO_FB *)tao->data;
  PetscReal gnorm0, gnorm, gradnorm;

  PetscFunctionBegin;

#if 0
  PetscCall(VecNorm(tao->gradient, NORM_2, &gradnorm));
  PetscCall(VecWAXPY(fb->workvec3, -1., tao->solution, fb->dualvec));
  PetscCall(VecScale(fb->workvec3, 1/tao->step));
  PetscCall(VecNorm(fb->workvec3, NORM_2, &gnorm0));
  PetscCall(VecAXPY(fb->workvec3, 1, tao->gradient));
  PetscCall(VecNorm(fb->workvec3, NORM_2, &gnorm));

  tao->gnorm0   = PetscMax(gradnorm, gnorm0) + tao->gttol;
  tao->residual = gnorm;
#endif
  if (!fb->use_accel && !fb->use_adapt && tao->linesearch->max_funcs == 0) {
    /* Fixed PG */
    PetscCall(VecWAXPY(fb->workvec3, -1., tao->solution, fb->dualvec));
    PetscCall(VecScale(fb->workvec3, 1/tao->step));
    PetscCall(VecAXPY(fb->workvec3, 1, tao->gradient));
    PetscCall(VecNorm(fb->workvec3, NORM_2, &gnorm));

    gnorm        *= gnorm;
    tao->residual = PetscSqrtReal(gnorm);
  } else if (!fb->use_accel && !fb->use_adapt && tao->linesearch->max_funcs > 0) {
    /* Backtracking PG */
    PetscCall(VecWAXPY(fb->workvec3, -1., tao->solution, fb->dualvec));
    PetscCall(VecNorm(fb->workvec3, NORM_2, &tao->residual));
    tao->residual /= tao->step;
  } else if (fb->use_accel) {
    /* Nesterov-types, both backtracking and fixed */
    PetscCall(VecWAXPY(fb->workvec3, -1., tao->solution, fb->dualvec));
    PetscCall(VecNorm(fb->workvec3, NORM_2, &tao->residual));
    tao->residual /= tao->step;
  } else if (fb->use_adapt) {
    PetscCall(VecWAXPY(fb->workvec3, -1., tao->solution, fb->dualvec));
    PetscCall(VecScale(fb->workvec3, 1/tao->step));
    PetscCall(VecAXPY(fb->workvec3, 1, tao->gradient));
    PetscCall(VecNorm(fb->workvec3, NORM_2, &gnorm));

    gnorm        *= gnorm;
    tao->residual = PetscSqrtReal(gnorm);
  } else PetscUnreachable();

  PetscCall(TaoLogConvergenceHistory(tao, f, tao->residual, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, tao->residual, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  tao->niter++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFB_ObjGrad_Private(Tao tao, Vec x, PetscReal *f, Vec g)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  if (fb->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, x, f, g));
    *f *= fb->f_scale;
    PetscCall(VecScale(g, fb->f_scale));
  } else {
    PetscCall(TaoComputeObjectiveAndGradient(tao, x, f, g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_FB(Tao tao)
{
  TAO_FB                      *fb = (TAO_FB *)tao->data;
  PetscReal                    f, f_prox;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");
  PetscCheck(fb->eta >= 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Backtracing scale factor needs to be equal or greater than 1");
  PetscCheck(!(fb->use_accel && fb->use_adapt), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "TaoFB only supports either acceleration or adaptive step, not both");
  if (!fb->lip_set) PetscCall(DMTaoGetLipschitz(fb->smoothterm, &fb->lip));


  if (fb->approx_lip || fb->lip == 0) {//TODO if lip is set, should we still approximate lip?
    /* Approximating initial Lipschitz number via two random vectors */
    PetscReal   gradnorm, xnorm;
    PetscRandom rctx;

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(fb->workvec, rctx));
    PetscCall(VecSetRandom(fb->workvec2, rctx));//TODO are they different? check later
    /* If TaoFBSetSmoothTerm has been called, will use DM for gradient.
     * Otherwise, use one set via TaoSetGradient */
    if (fb->smoothterm) {
      PetscCall(DMTaoComputeGradient(fb->smoothterm, fb->workvec, fb->x_old));
      PetscCall(DMTaoComputeGradient(fb->smoothterm, fb->workvec2, fb->grad_old));
    } else {
      PetscCall(TaoComputeGradient(tao, fb->workvec, fb->x_old));
      PetscCall(TaoComputeGradient(tao, fb->workvec2, fb->grad_old));
    }
    PetscCall(VecAXPY(fb->grad_old, -1., fb->x_old));
    PetscCall(VecAXPY(fb->workvec, -1., fb->workvec2));
    PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
    gradnorm *= fb->f_scale;
    PetscCall(VecNorm(fb->workvec, NORM_2, &xnorm));

    fb->lip   = gradnorm / xnorm;
    fb->lip   = PetscMax(fb->lip, 1.e-6);
    tao->step = 2. / fb->lip / 10.;

    PetscCall(PetscRandomDestroy(&rctx));
  } else if (fb->lip > 0) {
    tao->step = 1. / fb->lip;//TODO what if lip is set but want to use some set init step? There is no TaoSetInitialStep, so maybe TaoPSSetIntialStep? or former suffices?
  }

  fb->step_old = tao->step;

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  if (fb->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
    f *= fb->f_scale;
    PetscCall(VecScale(tao->gradient, fb->f_scale));
  } else {
    PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  }

  tao->reason = TAO_CONTINUE_ITERATING;

  if (fb->use_accel) {
    PetscCall(PetscCitationsRegister(fasta_citation, &fasta_cited));
    /* FISTA: dualvec = z_accel_1, workvec = z_accel_0 */
    PetscCall(VecCopy(tao->solution, fb->dualvec));
  }
  if (fb->use_adapt) {
    /* TODO initial stepsize update) */
    PetscCall(PetscCitationsRegister(adapgm_citation, &adapgm_cited));
  }

//TODO fixed nesterov, muf mug stuff
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscCall(VecCopy(tao->solution, fb->x_old));
    if (fb->use_adapt) PetscCall(VecCopy(tao->gradient, fb->grad_old));

    /* Backtrackig PG stepsize scaling */
    if (!fb->use_accel && !fb->use_adapt && tao->linesearch->max_funcs > 0) tao->step *= fb->eta;

    /* Note: DMTaoApplyProximalMap's scale is 1/(2*step) */
    if (fb->use_accel || (!fb->use_accel && !fb->use_adapt && tao->linesearch->max_funcs > 0)) {
      /* Nesterov-type, and backtracking PG */
      PetscCall(VecWAXPY(fb->workvec, -tao->step, tao->gradient, fb->dualvec));
      PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, tao->step*fb->prox_scale, fb->workvec, tao->solution, PETSC_FALSE));
      tao->nproxs++;
    } else {
      /* Fixed PG, and adaPGM */
      PetscCall(VecWAXPY(fb->dualvec, -tao->step, tao->gradient, tao->solution));
      PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, tao->step*fb->prox_scale, fb->dualvec, tao->solution, PETSC_FALSE));
      tao->nproxs++;
    }

    /* -tao_ls_max_funcs 0 -> no linesearch, but constant stepsize
       In this case, constant stepsize needs to be properly chosen for the algorithm to converge.
       TODO not so sure aboutt theory about nonmonotonic for nesterov-type
       -tao_ls_PSArmijo_memory_size  1 -> monotonic linesearch
       -tao_ls_PSArmijo_memory_size >1 -> nonmonotonic linesearch */
    if (!fb->use_adapt && tao->linesearch->max_funcs > 0) {
      //TODO not sure whether this works for backtracking nesterov
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
      PetscCall(TaoLineSearchApply(tao->linesearch, fb->dualvec, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));
      /* Linesearch failure. Abort */
      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->step   = 0.;
        tao->reason = TAO_DIVERGED_LS_FAILURE;
      }
      PetscCall(TaoLineSearchGetStepLength(tao->linesearch, &tao->step));
    }

    /* Post-processings */
    /* Fixed PGM  and adaPGM */
    if (fb->use_adapt || (!fb->use_accel && tao->linesearch->max_funcs == 0)) PetscCall(TaoFB_ObjGrad_Private(tao, tao->solution, &f, tao->gradient));

    PetscCall(DMTaoComputeObjective(fb->proxterm, tao->solution, &f_prox));
    f_prox *= fb->prox_scale;
    PetscCall(TaoFB_ComputeResidual_And_LogConv_Private(tao, f+f_prox));

    if (!fb->use_accel && !fb->use_adapt && tao->linesearch->max_funcs > 0) {
      /* backtracking PGM */
      PetscCall(VecCopy(tao->solution, fb->dualvec));
      PetscCall(TaoFB_ObjGrad_Private(tao, fb->dualvec, &f, tao->gradient));
     } else if (fb->use_accel) {
      /* Nesterov-type */
      fb->t_fista_old = fb->t_fista;
      fb->t_fista     = (1. + PetscSqrtReal(1. + 4. * fb->t_fista_old * fb->t_fista_old)) / 2.;
      fb->fista_beta  = (fb->t_fista_old - 1) / (fb->t_fista);

      /* z = x + beta(x - x_old) */
      PetscCall(VecAXPBYPCZ(fb->dualvec, 1. + fb->fista_beta, -fb->fista_beta, 0., tao->solution, fb->x_old));
      /* Now update f and grad */
      PetscCall(TaoFB_ObjGrad_Private(tao, fb->dualvec, &f, tao->gradient));
     } else if (fb->use_adapt) {
       PetscCall(TaoFB_ADAPGM_Update_Stepsize_Private(tao));
     }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_FB(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(x), where you have gradient of f(x), and proximal operator of g(x).");
  PetscCall(PetscOptionsReal("-tao_fb_initial_step", "Initial stepsize for forward-backward algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsReal("-tao_fb_bb_param", "Threshold parameter for Barzila-Borwein  stepsize rule for SPARSA algorithm", "", fb->bb_param, &fb->bb_param, NULL));
  PetscCall(PetscOptionsReal("-tao_fb_ls_scale", "Scaling parameter for backtracking proximal gradient", "", fb->eta, &fb->eta, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_approx_lip", "Approximate Lipschitz in the beginning", "", fb->approx_lip, &fb->approx_lip, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_accel", "Use Acceleration (Nesterov-type)", "", fb->use_accel, &fb->use_accel, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_adaptive", "Use adaptive stepsize (adaPGM)", "", fb->use_adapt, &fb->use_adapt, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_FB(Tao tao, PetscViewer viewer)
{
  PetscBool isascii;
  TAO_FB   *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (fb->smoothterm) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Smooth Term:\n"));
    PetscCall(DMTaoView(fb->smoothterm, viewer));
    }
    if (fb->proxterm) {PetscCall(PetscViewerASCIIPrintf(viewer, "Proximal Term:\n"));
      PetscCall(DMTaoView(fb->proxterm, viewer));
    }
    //TODO do i need view for reg? prob not now...
    if (fb->reg) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Regularizer Term:\n"));
      PetscCall(DMTaoView(fb->reg, viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  if (!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if (!fb->workvec) PetscCall(VecDuplicate(tao->solution, &fb->workvec));
  if (!fb->workvec2) PetscCall(VecDuplicate(tao->solution, &fb->workvec2));
  if (!fb->workvec3) PetscCall(VecDuplicate(tao->solution, &fb->workvec3));
  if (!fb->dualvec) PetscCall(VecDuplicate(tao->solution, &fb->dualvec));
  if (!fb->x_old) PetscCall(VecDuplicate(tao->solution, &fb->x_old));
  if (!fb->grad_old) PetscCall(VecDuplicate(tao->solution, &fb->grad_old));
  if (!fb->s_vec_lv) {
    PetscCall(VecDuplicate(tao->solution, &fb->s_vec_lv)); //TODO dont need this for fista,sparsa
    PetscCall(VecCopy(tao->solution, fb->s_vec_lv));
  }
  //TODO option to set regularizer type??
  //if L is set, should default stepsize be that? not for sparsa...
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &fb->reg));
  PetscCall(DMTaoSetType(fb->reg, DMTAOL2));
  if (fb->smoothterm) {
    PetscCall(TaoLineSearchUseDM(tao->linesearch, fb->smoothterm));
  } else {
    PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  }
  PetscCall(TaoLineSearchSetProx(tao->linesearch, fb->proxterm, fb->prox_scale, fb->reg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&fb->s_vec_lv));
  PetscCall(VecDestroy(&fb->workvec));
  PetscCall(VecDestroy(&fb->workvec2));
  PetscCall(VecDestroy(&fb->workvec3));
  PetscCall(VecDestroy(&fb->dualvec));
  PetscCall(VecDestroy(&fb->dualwork));
  PetscCall(VecDestroy(&fb->x_old));
  PetscCall(VecDestroy(&fb->grad_old));
  PetscCall(DMDestroy(&fb->reg));
  PetscCall(DMDestroy(&fb->smoothterm));
  PetscCall(DMDestroy(&fb->proxterm));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_FB(Tao tao)
{
  TAO_FB     *fb;
  const char *armijo_type = TAOLINESEARCHPSARMIJO;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fb));


  tao->gttol = 1.e-8;

  tao->ops->destroy           = TaoDestroy_FB;
  tao->ops->setup             = TaoSetUp_FB;
  tao->ops->setfromoptions    = TaoSetFromOptions_FB;
  tao->ops->view              = TaoView_FB;
  tao->ops->solve             = TaoSolve_FB;
//  tao->ops->convergencetest   = TaoConvergenceTest_FB;
  tao->ops->convergencetest   = TaoDefaultConvergenceTest;

  tao->data = (void *)fb;

  fb->t_fista     = 1;
  fb->t_fista_old = 1;
  fb->fista_beta  = 0.;
  fb->mu_fg       = 0.;
  fb->gnorm_norm  = 0.;
  fb->bb_param    = 0.5;
  fb->eta         = 1.;
  fb->smoothterm  = NULL;
  fb->proxterm    = NULL;
  fb->lip_set     = PETSC_FALSE;
  fb->approx_lip  = PETSC_TRUE;
  fb->use_accel   = PETSC_TRUE;
  fb->use_adapt   = PETSC_FALSE;

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
