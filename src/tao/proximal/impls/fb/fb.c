#include  <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

const char *const TaoFBTypes[] = {"type_fista", "type_sparsa", "type_lv", "TaoFBType", "TAO_FB_", NULL};

static PetscErrorCode TaoFBGetType_FB(Tao tao, TaoFBType *type)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  *type = fb->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoFBSetType_FB(Tao tao, TaoFBType type)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  fb->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBSetType(Tao tao, TaoFBType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, type, 2);
  PetscTryMethod(tao, "TaoFBSetType_C", (Tao, TaoFBType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBGetType(Tao tao, TaoFBType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoFBGetType_C", (Tao, TaoFBType *), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBSetLipschitz(Tao tao, PetscReal lip)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lip, 2);
  fb->lip = lip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBGetLipschitz(Tao tao, PetscReal *lip)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  *lip = fb->lip;
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
  fb->lmap     = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode TaoFBSetNonSmoothTermWithLinearMap(Tao tao, DM dm, Mat mat)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  if (mat) {
    PetscValidHeaderSpecific(mat, MAT_CLASSID, 3);
    PetscCall(PetscObjectReference((PetscObject)mat));
  }
  fb->proxterm = dm;
  fb->lmap = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoFBSetSmoothTerm(Tao tao, DM dm)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)dm));
  fb->smoothterm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoConvergenceTest_FB(Tao tao, void *dummy)
{
  TAO_FB            *fb = (TAO_FB *)tao->data;
  PetscInt           niter = tao->niter, nfuncs = PetscMax(tao->nfuncs, tao->nfuncgrads);
  PetscInt           max_funcs = tao->max_funcs;
  PetscReal          gnorm = tao->residual, gnorm0 = tao->gnorm0;
  PetscReal          f = tao->fc;
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

/* Fixed Point Characterization of the problem.
 * Essentially solves the problem for only one iteration.
 * Needed by linesearch routine
 *
 * out = prox_g(in - step*grad_f(in).
 *
 * we are give in vector, but we only need in-step*grad_f(in)...
 *
 * Should we assume that LS obj will give in-step*gradf(in)?
 * or stash this vec inside tao obj??
 *
 * Not entirely sure what this means for LV.
 * Dont think there is robust theory about adaptive stepsize
 * or backtracking for LV...  */
static PetscErrorCode TaoFixedPoint_FB(Tao tao, Vec in, Vec out, PetscReal step, Mat vm, void *ctx)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  switch (fb->type) {
  case TAO_FB_TYPE_FISTA:
  case TAO_FB_TYPE_SPARSA:
    PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, step, in, out, PETSC_FALSE, NULL));
    break;
  case TAO_FB_TYPE_LV:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Not supporting fixed-point iteration for LV type..");
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_FB(Tao tao)
{
  TAO_FB                      *fb = (TAO_FB *)tao->data;
  PetscReal                    f, temp, temp2, gnorm;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;

  PetscFunctionBegin;

  PetscCheck(fb->lip >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Lipschitz constant cannot be negative");
  PetscCheck(tao->step >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize cannot be negative");

  if (fb->approx_lip || !fb->lip_set) {
    PetscReal   gradnorm, xnorm;
    PetscRandom rctx;

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    // Structure from FASTA paper, Goldstein et al.
    PetscCall(VecSetRandom(fb->workvec, rctx));
    PetscCall(VecSetRandom(fb->workvec2, rctx));
    PetscCall(TaoComputeGradient(tao, fb->workvec, fb->x_old));
    PetscCall(TaoComputeGradient(tao, fb->workvec2, fb->grad_old));
    PetscCall(VecAXPY(fb->grad_old, -1., fb->x_old));
    PetscCall(VecAXPY(fb->workvec, -1., fb->workvec2));
    PetscCall(VecNorm(fb->grad_old, NORM_2, &gradnorm));
    PetscCall(VecNorm(fb->workvec, NORM_2, &xnorm));

    fb->lip   = gradnorm/xnorm;
    fb->lip   = PetscMax(fb->lip, 1.e-6);
    tao->step = 2./fb->lip/10.;

    PetscCall(PetscRandomDestroy(&rctx));
  } else if (fb->lip_set) {
    tao->step = 1./fb->lip;
  }

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
  PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm));
  PetscCheck(!PetscIsInfOrNanReal(f) && !PetscIsInfOrNanReal(gnorm), PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
  PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, tao->step));
  PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  tao->reason = TAO_CONTINUE_ITERATING;

  switch (fb->type) {
  case TAO_FB_TYPE_FISTA:
  case TAO_FB_TYPE_SPARSA:
    PetscCall(VecCopy(tao->solution, fb->dualvec));
    break;
  case TAO_FB_TYPE_LV:
    //s_0 = x_0 + step *L^T u_0. s = workvec, a = workvec2
    PetscCall(MatMultTranspose(fb->lmap, fb->dualvec, fb->s_vec_lv));
    PetscCall(VecAYPX(fb->s_vec_lv, tao->step, tao->solution));
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid forward-backward type.");
  }

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscCall(VecCopy(tao->solution, fb->x_old));
    PetscCall(VecCopy(tao->gradient, fb->grad_old));
  //dualvec:
  //For FISTA and SPARSA, it is input for prox at the beginning of iter,
  //and accelerated/momentum'd output at the end of iter
  //For LV, it is u vector.
  /* Duplicate switch statement, as FISTA and SPARSA share
   * large body of code                                     */
  switch (fb->type) {
  case TAO_FB_TYPE_FISTA:
  case TAO_FB_TYPE_SPARSA:
    PetscCall(VecAXPY(fb->dualvec, -tao->step, tao->gradient));
    PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, tao->step, fb->dualvec, tao->solution, PETSC_FALSE, NULL));

    /* -tao_ls_max_funcs 0 -> no linesearch, but constant stepsize
       Constant stepsize needs to be properly chosen for the algo to converge.
       -tao_ls_max_funcs 1 -> monotonic linesearch
       -tao_ls_max_funcs >1 -> nonmonotonic linesearch, and size is set via -tao_ls_PSArmijo_memory_size              */
    if (tao->linesearch->max_funcs !=0) { /*technically this can be done inside LS, but this avoids unncessary func call stacks */
      PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
      PetscCall(TaoLineSearchApply(tao->linesearch, fb->x_old, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
      PetscCall(TaoAddLineSearchCounts(tao));
      if (ls_status != TAOLINESEARCH_SUCCESS) {
        //TODO is resetting to 1 the best choice?
        PetscInfo(tao, "Line search failed at iteration %" PetscInt_FMT ": resetting stepsize to 1.\n", tao->niter);
        tao->step = 1.;
      }
    }
    PetscCall(TaoLineSearchGetStepLength(tao->linesearch, &tao->step));

    PetscCall(VecWAXPY(fb->workvec, -1., fb->x_old, tao->solution));
    PetscCall(VecWAXPY(fb->workvec2, -1., tao->solution, fb->dualvec));
    PetscCall(VecNorm(fb->workvec, NORM_2, &gnorm));
    gnorm /= tao->step;
    PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &temp));
    PetscCall(VecNorm(fb->workvec2, NORM_2, &temp2));
    fb->gnorm_norm = PetscMax(temp, temp2/tao->step) + tao->grtol;

    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);

    PetscCall(TaoComputeObjectiveAndGradient(tao, fb->dualvec, &f, tao->gradient));
    break;
  case TAO_FB_TYPE_LV:
    // workvec2: 2x_k - step*gradf(x_k) - s_k
    PetscCall(VecAXPBYPCZ(fb->workvec2, 2, -tao->step, 0, tao->solution, tao->gradient));
    PetscCall(VecAXPY(fb->workvec2, -1., fb->s_vec_lv));
    // s =  s + rho * a
    PetscCall(VecAXPBYPCZ(fb->s_vec_lv, fb->rho_lv, -fb->rho_lv*tao->step, 1-fb->rho_lv, tao->solution, tao->gradient));
    PetscCall(MatMult(fb->lmap, fb->workvec2, fb->workvec));
    PetscCall(VecScale(fb->workvec, fb->sigma_lv));
    PetscCall(VecAXPY(fb->workvec, 1., fb->dualvec));

    PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, fb->sigma_lv, fb->workvec, fb->workvec2, PETSC_TRUE, NULL));
    PetscCall(VecAXPBY(fb->dualvec, fb->rho_lv, 1-fb->rho_lv, fb->workvec));

    PetscCall(MatMultTranspose(fb->lmap, fb->dualvec, fb->workvec));
    PetscCall(VecWAXPY(tao->solution, -tao->step, fb->workvec, fb->s_vec_lv));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid forward-backward type.");
  }


  /* Adaptive stepsize, and acceleration part */
  switch (fb->type) {
  case TAO_FB_TYPE_FISTA:
    /* Restart FISTA */
    PetscCall(VecTDot(fb->workvec, fb->workvec2, &temp));

    fb->t_fista_old = (temp>0) ? 1 : fb->t_fista;
    fb->t_fista     = (1. + PetscSqrtReal(1.+4.*fb->t_fista*fb->t_fista)) /2.;
    temp            = 1. + (fb->t_fista_old -1)/(fb->t_fista);

    PetscCall(VecAXPBYPCZ(fb->dualvec, -1., temp, 0., tao->solution, fb->x_old));
    break;
  case TAO_FB_TYPE_SPARSA:
  {
    PetscReal snorm, sfnorm, bb_1, bb_2, bb;
    /* Adaptive stepsize. Will use Barzila-Borwein for default */

    PetscCall(VecWAXPY(fb->workvec2, -1., fb->x_old, fb->dualvec));
    PetscCall(VecAYPX(fb->workvec2, 1/tao->step, tao->gradient));
    PetscCall(VecTDot(fb->workvec, fb->workvec2, &temp));
    PetscCall(VecNorm(fb->workvec, NORM_2, &snorm));
    PetscCall(VecNorm(fb->workvec2, NORM_2, &sfnorm));

    bb_1 = (snorm*snorm) / temp;
    bb_2 = temp / (sfnorm*sfnorm);
    bb_2 = PetscMax(bb_2,0);
    bb   = (2*bb_2 > bb_1) ?  bb_2 : bb_1 - fb->bb_param*bb_2;

    if (bb <= 0 || PetscIsInfOrNanReal(bb)) bb = tao->step*1.5;
  }
  //case TAO_FB_TYPE_VMPG_BB: //Park et al
    break;
  case TAO_FB_TYPE_LV:
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid forward-backward type.");
  }
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBGetType_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_FB(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Forward backward problem that solves f(x)+g(x), where you have gradient of f(x), and proximal operator of g(x).");
  PetscCall(PetscOptionsReal("-tao_fb_initial_step", "Initial stepsize for forward-backward algorithm", "", tao->step, &tao->step, NULL));
  PetscCall(PetscOptionsReal("-tao_fb_bb_param", "Threshold parameter for Barzila-Borwein  stepsize rule for SPARSA algorithm", "", fb->bb_param, &fb->bb_param, NULL));
  PetscCall(PetscOptionsBool("-tao_fb_approx_lip", "Approximate Lipschitz in the beginning", "", fb->approx_lip, &fb->approx_lip, NULL));
  PetscCall(TaoLineSearchSetFromOptions(tao->linesearch));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_FB(Tao tao, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  if(!tao->gradient) PetscCall(VecDuplicate(tao->solution, &tao->gradient));
  if(!fb->workvec) PetscCall(VecDuplicate(tao->solution, &fb->workvec));
  if(!fb->workvec2)PetscCall(VecDuplicate(tao->solution, &fb->workvec2));
  if(!fb->dualvec) PetscCall(VecDuplicate(tao->solution, &fb->dualvec));
  if(!fb->x_old) PetscCall(VecDuplicate(tao->solution, &fb->x_old));
  if(!fb->grad_old) PetscCall(VecDuplicate(tao->solution, &fb->grad_old));
  if(!fb->s_vec_lv) PetscCall(VecDuplicate(tao->solution, &fb->s_vec_lv));//TODO dont need this for fista,sparsa
  //TODO option to set regularizer type??
  //if L is set, should default stepsize be that? not for sparsa...
  PetscCall(DMCreate(PetscObjectComm((PetscObject)tao), &fb->reg));
  PetscCall(DMTaoSetType(fb->reg, DMTAOL2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_FB(Tao tao)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBGetType_C", NULL));
  PetscCall(VecDestroy(&fb->s_vec_lv));
  PetscCall(VecDestroy(&fb->workvec));
  PetscCall(VecDestroy(&fb->workvec2));
  PetscCall(VecDestroy(&fb->dualvec));
  PetscCall(VecDestroy(&fb->x_old));
  PetscCall(VecDestroy(&fb->grad_old));
  PetscCall(DMDestroy(&fb->reg));
  //TODO do i need these here?
  PetscCall(DMDestroy(&fb->smoothterm));
  PetscCall(DMDestroy(&fb->proxterm));
  PetscCall(MatDestroy(&fb->lmap));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_FB(Tao tao)
{
  TAO_FB *fb;
  const char *armijo_type = TAOLINESEARCHARMIJO;

  PetscFunctionBegin;
  PetscCall(PetscNew(&fb));

  tao->ops->destroy           = TaoDestroy_FB;
  tao->ops->setup             = TaoSetUp_FB;
  tao->ops->setfromoptions    = TaoSetFromOptions_FB;
  tao->ops->view              = TaoView_FB;
  tao->ops->solve             = TaoSolve_FB;
  tao->ops->computefixedpoint = TaoFixedPoint_FB;
  tao->ops->convergencetest   = TaoConvergenceTest_FB;

  tao->data = (void *)fb;

  fb->t_fista     = 1;
  fb->t_fista_old = 1;
  fb->lip         = 0;
  fb->mu_f        = 0;
  fb->gnorm_norm  = 0.;
  fb->bb_param    = 0.5;
  fb->type        = TAO_FB_TYPE_FISTA;
  fb->lmap        = NULL;
  fb->smoothterm  = NULL;
  fb->proxterm    = NULL;
  fb->lip_set     = PETSC_FALSE;
  fb->mu_set      = PETSC_FALSE;
  fb->approx_lip  = PETSC_TRUE;

  /* Non-monotonic linesearch
   *
   * \hat{f}_k = max(f_{k-1},f_{k-2}, ... , f_{k-min(M,k)}), where M is linesearch history size
   * f(x_{k+1}) < \hat{f}_k + \langle x_{k+1} - x_k, grad_f(x_k) \rangle + (1/(2 step)) |x_{k+1} - x_k |^2
   *
   */
  PetscCall(TaoLineSearchCreate(PetscObjectComm((PetscObject)tao), &tao->linesearch));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1));
  PetscCall(TaoLineSearchSetType(tao->linesearch, armijo_type));
  PetscCall(TaoLineSearchUseTaoRoutines(tao->linesearch, tao));
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBSetType_C", TaoFBSetType_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBGetType_C", TaoFBGetType_FB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
