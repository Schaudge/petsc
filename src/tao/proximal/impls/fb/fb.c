#include <../src/tao/proximal/impls/fb/fb.h> /*I "petsctao.h" I*/
#include <petsctao.h>
#include <petsctaolinesearch.h>
#include <petscdm.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>

const char *const TaoFBTypes[] = {"fista", "sparsa", "lv", "TaoFBType", "TAO_FB_", NULL};

static PetscBool  cited      = PETSC_FALSE;
static const char citation[] = "@article{goldstein2015fasta,\n"
                               "title={FASTA: A generalized implementation of forward-backward splitting},\n"
                               "author={Goldstein, Tom and Studer, Christoph and Baraniuk, Richard},\n"
                               "journal={arXiv preprint arXiv:1501.04979},\n"
                               " year={2015}\n"
                               "} \n";

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

/*@
  TaoFBSetUseLipApprox - Determine whether to compute initial Lipschitz
  constant approximation or not.

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` solver
- use - Bool to denote whether to compute initial approximiation

  Level: advanced

.seealso: `Tao`, `TAOFB`
@*/
PetscErrorCode TaoFBSetUseLipApprox(Tao tao, PetscBool flag)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveBool(tao, flag, 2);
  fb->approx_lip = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoFBSetType - Determine the forward-backward algorithm type.

  Input Parameters:
+ tao  - the `Tao` context for the `TAOFB` solver
- type - forward-backward algorithm type

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TaoFBGetType()`, `TaoFBType`
@*/
PetscErrorCode TaoFBSetType(Tao tao, TaoFBType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(tao, type, 2);
  PetscTryMethod(tao, "TaoFBSetType_C", (Tao, TaoFBType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoFBGetType - Retrieve the forward-backward type.

  Input Parameter:
. tao - the `Tao` context for the `TAOFB` solver

  Output Parameter:
. type - forward-backward algorithm type

  Level: advanced

.seealso: `Tao`, `TAOFB`, `TaoFBSetType()`, `TaoFBType`
@*/
PetscErrorCode TaoFBGetType(Tao tao, TaoFBType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoFBGetType_C", (Tao, TaoFBType *), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoFBSetLipschitz - Set user-defined Lipschitz constant of the gradient term.
  Users can use this routine if the smooth term is set via `Tao` object.
  If the smooth term is set via `DMTao` object, then users can use
  `DMTaoSetLipschitz()` routine.

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` solver
- lip - Lipschitz constant of gradient term

  Level: advanced

.seealso: `TAOFB`, `Tao`, `TaoFBGetLipschitz()`
@*/
PetscErrorCode TaoFBSetLipschitz(Tao tao, PetscReal lip)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidLogicalCollectiveReal(tao, lip, 2);
  fb->lip     = lip;
  fb->lip_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoFBSetNonSmoothTerm - Set non-smooth objective term for
  forward-backward problem - h(x).

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` solver
- dm  - the `DMTao` context containing non-smooth objective

  Level: advanced

.seealso: `TAOFB`, `Tao`, `TaoFBSetNonSmoothTermWithLinearMap()`
@*/
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

/*@
  TaoFBSetNonSmoothTermWithLinearMap - Set non-smooth objective term
  with linear mapping for forward-backward problem - g(Ax).

  Input Parameters:
+ tao  - the `Tao` context for the `TAOFB` solver
. dm   - the `DMTao` context containing non-smooth objective
. mat  - the linear mapping matrix
- norm - norm of the linear mapping matrix, if avaliable. Set as zero if unknown

  Level: advanced

.seealso: `TAOFB`, `Tao`, `TaoFBSetNonSmoothTerm()`
@*/
PetscErrorCode TaoFBSetNonSmoothTermWithLinearMap(Tao tao, DM dm, Mat mat, PetscReal norm)
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
  PetscValidLogicalCollectiveReal(tao, norm, 4);
  fb->proxterm  = dm;
  fb->lmap      = mat;
  fb->lmap_norm = norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoFBSetSmoothTerm - Set smooth objective term for forward-backward problem.
  This smooth objective - f(x) -  must have gradient term available.

  Input Parameters:
+ tao - the `Tao` context for the `TAOFB` solver
- dm  - the `DMTao` context containing smooth objective

  Level: advanced

.seealso: `TAOFB`, `Tao`, `TaoFBSetNonSmoothTerm()`
@*/
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

/* Fixed Point Characterization of the problem.
 * Essentially solves the problem for only one iteration.
 * Needed by linesearch routine
 *
 * out = prox_g(in - step*grad_f(in)).
 *
 * For FISTA and SPARSA, input vector is assumed to be input for proximal mapping,
 * i.e., in := x - step*gradf(x), to avoid unncessary computation.                 */
static PetscErrorCode TaoFixedPoint_FB(Tao tao, Vec in, Vec out, PetscReal step, Mat vm, void *ctx)
{
  TAO_FB *fb = (TAO_FB *)tao->data;

  PetscFunctionBegin;
  switch (fb->type) {
  case TAO_FB_FISTA:
  case TAO_FB_SPARSA:
    PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, step, in, out, PETSC_FALSE));
    break;
  case TAO_FB_LV:
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

  switch (fb->type) {
  case TAO_FB_FISTA:
  case TAO_FB_SPARSA:
  {
    if (fb->approx_lip || !fb->lip_set) {
      PetscReal   gradnorm, xnorm;
      PetscRandom rctx;

      PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
      PetscCall(PetscRandomSetFromOptions(rctx));
      /* Structure from FASTA paper, Goldstein et al. */
      PetscCall(VecSetRandom(fb->workvec, rctx));
      PetscCall(VecSetRandom(fb->workvec2, rctx));
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
      PetscCall(VecNorm(fb->workvec, NORM_2, &xnorm));

      fb->lip   = gradnorm / xnorm;
      fb->lip   = PetscMax(fb->lip, 1.e-6);
      tao->step = 2. / fb->lip / 10.;

      PetscCall(PetscRandomDestroy(&rctx));
    } else if (fb->lip_set) {
      tao->step = 1. / fb->lip;
    }
  }
    break;
  case TAO_FB_LV:
    tao->step = 0.99*2/1.53395234e-04;//TODO lv example hardcode
    fb->sigma_lv = 1/(tao->step*4);
    break;
  default:
    break;
  }

  fb->step_old = tao->step;

  PetscCheck(tao->step > 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Stepsize has to be greater than zero");
  if (fb->smoothterm) {
    PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
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

  switch (fb->type) {
  case TAO_FB_FISTA:
  case TAO_FB_SPARSA:
    PetscCall(VecCopy(tao->solution, fb->dualvec));
    break;
  case TAO_FB_LV:
    /*s_0 = x_0 + step *L^T u_0. s = workvec, a = workvec2,
     * but u is initialized as 0, and s as x, thus omit */
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid forward-backward type.");
  }

  PetscCall(PetscCitationsRegister(citation, &cited));

  while (tao->reason == TAO_CONTINUE_ITERATING) {
    PetscCall(VecCopy(tao->solution, fb->x_old));
    PetscCall(VecCopy(tao->gradient, fb->grad_old));

    /* Duplicate switch statement, as FISTA and SPARSA share large body of code */
    switch (fb->type) {
    case TAO_FB_FISTA:
    case TAO_FB_SPARSA:
      PetscCall(VecWAXPY(fb->workvec2, -tao->step, tao->gradient, tao->solution));
      PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, tao->step, fb->workvec2, tao->solution, PETSC_FALSE));

      /* -tao_ls_max_funcs 0 -> no linesearch, but constant stepsize
       Constant stepsize needs to be properly chosen for the algorithm to converge.
       -tao_ls_max_funcs 1 -> monotonic linesearch
       -tao_ls_max_funcs >1 -> nonmonotonic linesearch, and size is set via -tao_ls_PSArmijo_memory_size */
      if (tao->linesearch->max_funcs != 0) {
        PetscCall(TaoLineSearchSetInitialStepLength(tao->linesearch, tao->step));
        PetscCall(TaoLineSearchApply(tao->linesearch, fb->x_old, &f, tao->gradient, tao->solution, &tao->step, &ls_status));
        PetscCall(TaoAddLineSearchCounts(tao));
        if (ls_status != TAOLINESEARCH_SUCCESS) {
          //TODO is resetting to 1 the best choice?
          PetscInfo(tao, "Line search failed at iteration %" PetscInt_FMT ": resetting stepsize to 1.\n", tao->niter);
          tao->step = 1.;
        }
        PetscCall(TaoLineSearchGetStepLength(tao->linesearch, &tao->step));
      }

      PetscCall(TaoGradientNorm(tao, tao->gradient, NORM_2, &temp));
      PetscCall(VecAXPY(fb->workvec2, -1., tao->solution));
      PetscCall(VecNorm(fb->workvec2, NORM_2, &temp2));
      fb->gnorm_norm = PetscMax(temp, temp2 / tao->step) + tao->grtol;
      break;
    case TAO_FB_LV:
      /* workvec2 = a: x_k - step*gradf(x_k) - s_k */
      PetscCall(VecAXPBYPCZ(fb->workvec2, 1, -tao->step, 0, tao->solution, tao->gradient));
      PetscCall(VecAXPY(fb->workvec2, -1., fb->s_vec_lv));
      /* s =  s + rho * a */
      PetscCall(VecAXPY(fb->s_vec_lv, fb->rho_lv, fb->workvec2));
      /* workvec2: x + a */
      PetscCall(VecAXPY(fb->workvec2, 1., tao->solution));
      /* dualwork : u + sigma*A(x+a) */
      PetscCall(MatMult(fb->lmap, fb->workvec2, fb->dualwork));
      PetscCall(VecScale(fb->dualwork, fb->sigma_lv));
      PetscCall(VecAXPY(fb->dualwork, 1., fb->dualvec));

      PetscCall(DMTaoApplyProximalMap(fb->proxterm, fb->reg, fb->sigma_lv, fb->dualwork, fb->dualwork2, PETSC_TRUE));
      PetscCall(VecAXPBY(fb->dualvec, fb->rho_lv, 1 - fb->rho_lv, fb->dualwork2));
      PetscCall(MatMultTranspose(fb->lmap, fb->dualvec, fb->workvec2));
      PetscCall(VecWAXPY(tao->solution, -tao->step, fb->workvec2, fb->s_vec_lv));
      //TODO linesearch for adaptive stepsize
      fb->gnorm_norm = 0.;//TODO ???
      break;
    default:
      SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Invalid forward-backward type.");
    }

    PetscCall(VecWAXPY(fb->workvec, -1., fb->x_old, tao->solution));
    PetscCall(VecNorm(fb->workvec, NORM_2, &gnorm));
    gnorm /= tao->step;
    tao->niter++;
    PetscCall(TaoLogConvergenceHistory(tao, f, gnorm, 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, f, gnorm, 0.0, tao->step));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);

    /* Adaptive stepsize, and acceleration part */
    switch (fb->type) {
    case TAO_FB_FISTA:
      /* Restart FISTA */
      /* workvec = xnew - xold, dualvec: x_accel1, workvec2: x_accel1 - x_accel0 */
      PetscCall(VecWAXPY(fb->workvec2, -1., fb->dualvec, tao->solution));
      PetscCall(VecCopy(tao->solution, fb->dualvec));
      /* workvec2 = x_accel1 - x_accel0 */
      PetscCall(VecTDot(fb->workvec, fb->workvec2, &temp));

      fb->t_fista_old = (temp < 0) ? 1 : fb->t_fista;
      fb->t_fista     = (1. + PetscSqrtReal(1. + 4. * fb->t_fista_old * fb->t_fista_old)) / 2.;
      temp            = (fb->t_fista_old - 1) / (fb->t_fista);

      PetscCall(VecWAXPY(tao->solution, temp, fb->workvec2, fb->dualvec));
      if (fb->smoothterm) {
        PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
      } else {
        PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
      }
      break;
    case TAO_FB_SPARSA: {
      /* Adaptive stepsize. Will use Barzila-Borwein for default */
      PetscReal snorm, sfnorm, bb_1, bb_2;

      if (fb->smoothterm) {
        PetscCall(DMTaoComputeObjectiveAndGradient(fb->smoothterm, tao->solution, &f, tao->gradient));
      } else {
        PetscCall(TaoComputeObjectiveAndGradient(tao, tao->solution, &f, tao->gradient));
      }
      /* workvec: diffx, workvec2: diffgrad */
      PetscCall(VecWAXPY(fb->workvec2, -1., fb->grad_old, tao->gradient));
      PetscCall(VecTDot(fb->workvec, fb->workvec2, &temp));
      PetscCall(VecNorm(fb->workvec, NORM_2, &snorm));
      PetscCall(VecNorm(fb->workvec2, NORM_2, &sfnorm));

      bb_1      = (snorm * snorm) / temp;
      bb_2      = temp / (sfnorm * sfnorm);
      bb_2      = PetscMax(bb_2, 0);
      tao->step = (2 * bb_2 > bb_1) ? bb_2 : bb_1 - fb->bb_param * bb_2;

      if (tao->step <= 0 || PetscIsInfOrNanReal(tao->step)) tao->step = fb->step_old * 1.5;
    }

    //CASE TAO_FB_BB3? Stabilized BB?
    //case TAO_FB_VMPG_BB: //Park et al
    //case TAO_FB_ADAPGM: Latafat etc..
    break;
    case TAO_FB_LV:
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
  PetscCall(PetscOptionsEnum("-tao_fb_type", "Forward-backward solver type", "TaoFBType", TaoFBTypes, (PetscEnum)fb->type, (PetscEnum *)&fb->type, NULL));
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
    PetscCall(PetscViewerASCIIPrintf(viewer, "FB Type: %s\n", TaoFBTypes[fb->type]));
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
    if (fb->lmap) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Linear Operator:\n"));
      PetscCall(MatView(fb->lmap, viewer));
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
  /* dualvec is same size as sol if lmap is not set.
   * If lmap is set, it is same as leftside of matrix */
  if (!fb->dualvec) {
    if (fb->lmap) {
      PetscCall(MatCreateVecs(fb->lmap, NULL, &fb->dualvec));
      PetscCall(MatCreateVecs(fb->lmap, NULL, &fb->dualwork));
      PetscCall(MatCreateVecs(fb->lmap, NULL, &fb->dualwork2));
      PetscCall(VecZeroEntries(fb->dualvec));
    } else {
      PetscCall(VecDuplicate(tao->solution, &fb->dualvec));
    }
  }
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
  PetscCall(VecDestroy(&fb->workvec3));
  PetscCall(VecDestroy(&fb->dualvec));
  PetscCall(VecDestroy(&fb->dualwork));
  PetscCall(VecDestroy(&fb->x_old));
  PetscCall(VecDestroy(&fb->grad_old));
  PetscCall(DMDestroy(&fb->reg));
  PetscCall(DMDestroy(&fb->smoothterm));
  PetscCall(DMDestroy(&fb->proxterm));
  PetscCall(MatDestroy(&fb->lmap));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_FB(Tao tao)
{
  TAO_FB     *fb;
  const char *armijo_type = TAOLINESEARCHPSARMIJO;

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
  fb->lmap_norm   = 0.;
  fb->sigma_lv    = 1.;
  fb->rho_lv      = 1.; //TODO whats optimal init?
  fb->type        = TAO_FB_FISTA;
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
  PetscCall(TaoLineSearchSetOptionsPrefix(tao->linesearch, tao->hdr.prefix));

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBSetType_C", TaoFBSetType_FB));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoFBGetType_C", TaoFBGetType_FB));
  PetscFunctionReturn(PETSC_SUCCESS);
}
