#include <petsc/private/taoimpl.h>
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/psarmijo/psarmijo.h>

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
.  xnew - search direction (x_{k+1})
.  f    - f(x_k)
.  g    - grad_f(x_k). same as tao->gradient
-  step - initial estimate of step length (Set via TaoLSSetInitialStep)

   Output parameters:
+  f    - f(x_{k+1})
.  xnew - x_{k+1} that satisfies the condition
-  step - final step length

   Info is set to one of:
.   0 - the line search succeeds; the sufficient decrease
   condition hold

   negative number if an input parameter is invalid
-   -1 -  step < 0

   positive number > 1 if the line search otherwise terminates
+    1 -  Step is at the lower bound, stepmin.
@ */
static PetscErrorCode TaoLineSearchApply_PSArmijo(TaoLineSearch ls, Vec xold, PetscReal *f, Vec g, Vec xnew)
{
  TaoLineSearch_PSARMIJO *armP = (TaoLineSearch_PSARMIJO *)ls->data;
  PetscInt                i, its = 0;
  PetscReal               ref, cert;
  PetscReal               diffnorm, inprod;
  PetscReal               fk1; /* f(x_{k+1}) */
  MPI_Comm                comm;

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
  /* work  : x_k - step*grad_f(x_k) */

  /* Calculate function at new iterate */
  PetscCall(TaoLineSearchComputeObjective(ls, xnew, &fk1));
  /* work2 : x_{k+1} - x_k */
  PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
  PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
  PetscCall(VecTDot(armP->work2, g, &inprod));
  cert = ref + inprod + (1 / (2 * ls->step)) * diffnorm - fk1;

  if (cert > ls->rtol) {
    ls->reason = TAOLINESEARCH_SUCCESS;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  while (cert <= ls->rtol && ls->nfeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;

    /* Note: usual eta: SPARSA:0.2, FISTA:0.5 */
    ls->step = ls->step * armP->eta;
    /* FPI for FB assumes x_k - step*gradf(x_k) as an input TODO fix for LV */
    PetscCall(VecWAXPY(armP->work, -ls->step, g, xold));
    PetscCall(TaoLineSearchComputeProximalMap(ls, ls->step, armP->work, xnew, PETSC_FALSE));

    /* Calculate function at new iterate */
    PetscCall(TaoLineSearchComputeObjective(ls, xnew, &fk1));
    /* work2 : x_{k+1} - x_k */
    PetscCall(VecWAXPY(armP->work2, -1., xold, xnew));
    PetscCall(VecTDot(armP->work2, armP->work2, &diffnorm));
    PetscCall(VecTDot(armP->work2, g, &inprod));
    cert = ref + inprod + (1 / (2 * ls->step)) * diffnorm - fk1;

    if (cert > ls->rtol) {
      ls->reason = TAOLINESEARCH_SUCCESS;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->nfeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search function evals (%" PetscInt_FMT ") > maximum allowed (%" PetscInt_FMT ")\n", ls->nfeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
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
