#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/psarmijo/psarmijo.h>
#include <../src/tao/proximal/impls/cv/cv.h>

static PetscErrorCode TaoLineSearchDestroy_PSArmijo(TaoLineSearch ls)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(armP->memory));
  if (armP->x) PetscCall(PetscObjectDereference((PetscObject)armP->x));
  PetscCall(VecDestroy(&armP->work));
  PetscCall(VecDestroy(&armP->work2));
  PetscCall(PetscFree(ls->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetFromOptions_PSArmijo(TaoLineSearch ls, PetscOptionItems *PetscOptionsObject)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PSArmijo linesearch options");
  PetscCall(PetscOptionsReal("-tao_ls_PSArmijo_eta", "decrease constant", "", armP->eta, &armP->eta, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_PSArmijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchView_PSArmijo(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;
  PetscBool               isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  PSArmijo linesearch"));
    PetscCall(PetscViewerASCIIPrintf(pv, "eta=%g ", (double)armP->eta));
    PetscCall(PetscViewerASCIIPrintf(pv, "memsize=%" PetscInt_FMT "\n", armP->memorySize));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* @ TaoApply_PSArmijo - This routine performs a linesearch. It
   backtracks until the (nonmonotone) PSArmijo conditions are satisfied.

   Input Parameters:
+  ls   - TaoLineSearch context
.  xold - x_k
.  f    - f(x_k)
.  g    - grad_f(x_k). same as tao->gradient
-  step - initial estimate of step length (Set via TaoLSSetInitialStep)

   Output parameters:
+  f    - f(x_{k+1})
.  xnew - x_{k+1} that satisfies the condition
-  step - final step length
@ */
static PetscErrorCode TaoLineSearchApply_PSArmijo(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;
  PetscInt                i, its = 0;
  PetscReal               ref, cert; /* cert = R + <gradf(x), xnew - x> + 1/2step * |xnew - x|_2^2 */
  PetscReal               diffnorm, inprod;
  PetscReal               fk1; /* f(xnew) */
  MPI_Comm                comm;
  TaoType                 type;
  PetscBool               isfb, iscv;
  PetscReal               xi, grad_x_dot, xdiffnorm, graddiffnorm, L, C, D, min1, min2, min3;
  PetscReal               temp, temp2, temp3, step_new, rho, norm1, norm2;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ls, &comm));
  ls->nfeval = 0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  } else if (xold != armP->x) {
    PetscCall(VecDestroy(&armP->work));
    PetscCall(VecDestroy(&armP->work2));
    PetscCall(VecDuplicate(xold, &armP->work));
    PetscCall(VecDuplicate(xold, &armP->work2));
    PetscCall(PetscObjectDereference((PetscObject)armP->x));
    armP->x = xold;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  }

  PetscCall(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  /* Check linesearch parameters */
  if (armP->eta > 1) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: eta (%g) > 1\n", (double)armP->eta));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: memory_size (%" PetscInt_FMT ") < 1\n", armP->memorySize));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "PSArmijo line search error: initial function inf or nan\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);


  PetscCall(TaoGetType(ls->tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)ls->tao, TAOFB, &isfb));
  PetscCall(PetscObjectTypeCompare((PetscObject)ls->tao, TAOCV, &iscv));

  //TODO maybe i should skip all the memory stuff if TAOCV since there is no non-monotonic theory about TAOCV
  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function values. */
  if (!armP->memory) PetscCall(PetscMalloc1(armP->memorySize, &armP->memory));

  if (!armP->memorySetup) {
    for (i = 0; i < armP->memorySize; i++) armP->memory[i] = (*f);
    armP->current     = 0;
    armP->memorySetup = PETSC_TRUE;
  }

  /* Calculate reference value (MAX) */
  ref = armP->memory[0];

  for (i = 1; i < armP->memorySize; i++) {
    if (armP->memory[i] > ref) { ref = armP->memory[i]; }
  }

  ls->step = ls->initstep;

  if (isfb) {
    // Input is prox_g(x- step * gradf(x))
    /* Calculate function at new iterate */
    PetscCall(TaoLineSearchComputeObjective(ls, xnew, &fk1));
    /* Check criteria */
    PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
    PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
    PetscCall(VecTDot(armP->work2, g, &inprod));
    cert = inprod + (1 / (2 * ls->step)) * diffnorm + ref;
    cert -= fk1;
    if (cert > ls->rtol) {
      ls->reason = TAOLINESEARCH_SUCCESS;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else if (iscv) {
    TAO_CV   *cv = (TAO_CV *)ls->tao->data;
    xi  = cv->pd_ratio * ls->tao->step * cv->g_lmap_norm;
    xi *= xi;

    PetscCall(VecWAXPY(cv->workvec, -1., cv->grad_old, ls->tao->gradient));
    PetscCall(VecWAXPY(cv->workvec2, -1., cv->x_old, ls->tao->solution));
    PetscCall(VecTDot(cv->workvec, cv->workvec2, &grad_x_dot));
    PetscCall(VecNorm(cv->workvec2, NORM_2, &xdiffnorm));
    PetscCall(VecNorm(cv->workvec, NORM_2, &graddiffnorm));

    L = grad_x_dot / (xdiffnorm*xdiffnorm);
    C = (graddiffnorm*graddiffnorm) / grad_x_dot;
    D = ls->tao->step* L * (ls->tao->step * C - 1);

    cv->g_lmap_norm *= cv->R;
    cert = PETSC_INFINITY; // For TAOCV, linesearch needs to go at least once
  } else SETERRQ(PetscObjectComm((PetscObject)ls->tao), PETSC_ERR_USER, "Invalid Tao type.");

  while (cert <= ls->rtol && ls->nfeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;

    if (isfb) {
      /* Note: usual eta: FISTA:0.5 */
      ls->step = ls->step * armP->eta;

      /* FB: input to prox: x_k - step*gradf(x_k)
       * Adaptive DY: input to prox: z - step*u - step*gradf(z) -> input g for DY needs to be u-gradf(z), not just gradf(z) TODO*/
      PetscCall(VecWAXPY(armP->work, -ls->step, g, xold));
      /* Note: DMTaoApplyProximalMap's step is for f(x)+step*g(x,y).
       * Thus, need pass stepsize as 1/2step                          */
      PetscCall(DMTaoApplyProximalMap(ls->prox, ls->prox_reg, 1/(2*ls->step), armP->work, xnew, PETSC_FALSE));
      /* Calculate function at new iterate */
      PetscCall(TaoLineSearchComputeObjective(ls, xnew, &fk1));
      /* work2 : x_{k+1} - x_k */
      PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
      PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
      PetscCall(VecTDot(armP->work2, g, &inprod));
      cert = ref + inprod + (1 / (2 * ls->step)) * diffnorm;
      cert -= fk1;

      if (cert > ls->rtol) {
        ls->reason = TAOLINESEARCH_SUCCESS;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    } else if (iscv) {
      TAO_CV   *cv = (TAO_CV *)ls->tao->data;

      min1      = ls->tao->step * PetscSqrtReal(1 + ls->tao->step / cv->step_old);
      min2      = 1 / (2 * cv->nu * cv->pd_ratio * cv->g_lmap_norm);
      temp      = 1 - 4*xi*(1+ls->tao->gatol)*(1+ls->tao->gatol);
      temp2     = cv->pd_ratio * cv->g_lmap_norm * ls->tao->step; //Unlike no linesearch, this "xi" uses updated norm estimate
      temp3     = PetscSqrtReal(D*D + temp*temp2*temp2);
      min3      = ls->tao->step * PetscSqrtReal(temp / (2*(1+ls->tao->gatol)*(temp3 +D)));
      step_new  = PetscMin(min1, PetscMin(min2, min3));
      rho       = step_new / ls->tao->step;
      cv->sigma = cv->pd_ratio*cv->pd_ratio*step_new;

      PetscCall(VecWAXPY(cv->dualvec_work, -cv->sigma * rho, cv->Ax_old, cv->dualvec));
      PetscCall(VecAXPY(cv->dualvec_work, cv->sigma*(1+rho), cv->Ax));
      PetscCall(DMTaoApplyProximalMap(cv->h_prox, cv->reg, 1/(2*cv->sigma), cv->dualvec_work, cv->dualvec_test, PETSC_TRUE));
      PetscCall(MatMultTranspose(cv->g_lmap, cv->dualvec_test, cv->workvec));
      /* norm1 = norm(ATy_test - ATy */
      PetscCall(VecWAXPY(cv->workvec2, -1., cv->ATy, cv->workvec));
      PetscCall(VecNorm(cv->workvec2, NORM_2, &norm1));
      /* norm2 = norm(y_test - y) */
      PetscCall(VecWAXPY(cv->dualvec_old, -1., cv->dualvec_test, cv->dualvec));
      PetscCall(VecNorm(cv->dualvec_old, NORM_2, &norm2));
      cert = cv->g_lmap_norm - norm1 / norm2;
      if (cv->g_lmap_norm >= norm1 / norm2) {
        cv->step_old  = ls->tao->step;
        ls->tao->step = step_new;
        PetscCall(VecCopy(cv->dualvec_test, cv->dualvec));
        PetscCall(VecCopy(cv->workvec, cv->ATy));
        ls->reason = TAOLINESEARCH_SUCCESS;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      cv->g_lmap_norm *= cv->r;
    } else SETERRQ(PetscObjectComm((PetscObject)ls->tao), PETSC_ERR_USER, "Invalid Tao type.");

    PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->nfeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search function evals (%" PetscInt_FMT ") > maximum allowed (%" PetscInt_FMT ")\n", ls->nfeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  } else if (ls->step < ls->stepmin) {
    PetscCall(PetscInfo(ls, "Step length is below tolernace.\n"));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  }
  if (ls->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Successful termination, update memory. Only FIFO for PSARMIJO */
  ls->reason                    = TAOLINESEARCH_SUCCESS;
  armP->memory[armP->current++] = *f;
  if (armP->current >= armP->memorySize) armP->current = 0;

  PetscCall(PetscInfo(ls, "%" PetscInt_FMT " function evals in line search, step = %10.4f\n", ls->nfeval, (double)ls->step));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   TAOLINESEARCHPSARMIJO - Special line-search type for proximal splittign algorithms.
   Should not be used with any other algorithm.

   Level: developer

seealso: `TaoLineSearch`, `TAOFB`, `Tao`
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_PSArmijo(TaoLineSearch ls)
{
  TaoLineSearch_PSARMIJO *armP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscNew(&armP));

  armP->memory            = NULL;
  armP->eta               = 0.5;
  armP->memorySize        = 1;
  ls->data                = (void *)armP;
  ls->initstep            = 0;
  ls->ops->monitor        = NULL;
  ls->ops->setup          = NULL;
  ls->ops->reset          = NULL;
  ls->ops->apply          = TaoLineSearchApply_PSArmijo;
  ls->ops->view           = TaoLineSearchView_PSArmijo;
  ls->ops->destroy        = TaoLineSearchDestroy_PSArmijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_PSArmijo;
  PetscFunctionReturn(PETSC_SUCCESS);
}
