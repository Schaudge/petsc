#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/vecimpl.h>

PetscClassId TAOTERM_CLASSID;

PetscLogEvent TAOTERM_ObjectiveEval;
PetscLogEvent TAOTERM_GradientEval;
PetscLogEvent TAOTERM_ObjGradEval;
PetscLogEvent TAOTERM_HessianEval;
PetscLogEvent TAOTERM_HessianMult;

const char *const TaoTermParametersModes[] = {"optional", "none", "required", "TaoTermParametersMode", "TAOTERM_PARAMETERS_", NULL};

/*@
  TaoTermDestroy - Destroy a `TaoTerm`.

  Collective

  Input Parameters:
. term - a `TaoTerm`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
@*/
PetscErrorCode TaoTermDestroy(TaoTerm *term)
{
  PetscFunctionBegin;
  if (!*term) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*term, TAOTERM_CLASSID, 1);
  if (--((PetscObject)*term)->refct > 0) {
    *term = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod(*term, destroy);
  PetscCall(PetscFree((*term)->H_mattype));
  PetscCall(PetscFree((*term)->Hpre_mattype));
  PetscCall(MatDestroy(&(*term)->solution_factory));
  PetscCall(MatDestroy(&(*term)->parameters_factory));
  PetscCall(MatDestroy(&(*term)->parameters_factory_orig));
  PetscCall(PetscHeaderDestroy(term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermView - View a description of a `TaoTerm`.

  Collective

  Input Parameters:
+ term   - a `TaoTerm`
- viewer - a `PetscViewer`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermView(TaoTerm term, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)term), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(term, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    const char *solution_vec_type;
    PetscInt    N;

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)term, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(MatGetVecType(term->solution_factory, &solution_vec_type));
    PetscCall(MatGetSize(term->solution_factory, &N, NULL));
    if (N < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: not set up yet [VecType %s (taoterm_solution_vec_type)]\n", solution_vec_type));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "solution vector space: N = %" PetscInt_FMT " [VecType %s (taoterm_solution_vec_type)]\n", N, solution_vec_type));
    if (term->parameters_mode == TAOTERM_PARAMETERS_NONE) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (none)\n"));
    } else {
      const char *parameters_vec_type;
      PetscInt    K;

      PetscCall(MatGetVecType(term->parameters_factory, &parameters_vec_type));
      PetscCall(MatGetSize(term->parameters_factory, &K, NULL));
      if (K < 0) PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (%s) not set up yet [VecType %s (taoterm_parameters_vec_type)]\n", TaoTermParametersModes[term->parameters_mode], parameters_vec_type));
      else PetscCall(PetscViewerASCIIPrintf(viewer, "parameter vector space: (%s) K = %" PetscInt_FMT " [VecType %s (taoterm_parameters_vec_type)]\n", TaoTermParametersModes[term->parameters_mode], K, parameters_vec_type));
    }
    if (term->ops->createhessianmatrices == TaoTermCreateHessianMatricesDefault) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian MatType (taoterm_hessian_mat_type): %s\n", term->H_mattype ? term->H_mattype : "(undefined)"));
      if (!term->Hpre_is_H) { PetscCall(PetscViewerASCIIPrintf(viewer, "default Hessian preconditioning MatType (taoterm_hessian_pre_mat_type): %s\n", term->Hpre_mattype ? term->Hpre_mattype : "(undefined)")); }
    }
    if (term->ops->view) { PetscUseTypeMethod(term, view, viewer); }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetUp - Set up a `TaoTerm`.

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermView()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermSetUp(TaoTerm term)
{
  PetscInt N;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (term->setup_called) PetscFunctionReturn(PETSC_SUCCESS);
  term->setup_called = PETSC_TRUE;
  PetscTryTypeMethod(term, setup);
  PetscCall(MatGetSize(term->solution_factory, &N, NULL));
  if (N < 0 && term->ops->createvecs) {
    Vec sol_template;

    PetscCall(TaoTermCreateVecs(term, &sol_template, NULL));
    PetscCall(TaoTermSetSolutionTemplate(term, sol_template));
    PetscCall(VecDestroy(&sol_template));
  }
  PetscCall(MatSetUp(term->solution_factory));
  if (term->parameters_mode != TAOTERM_PARAMETERS_NONE) {
    PetscInt K;

    PetscCall(MatGetSize(term->parameters_factory, &K, NULL));
    if (K < 0 && term->ops->createvecs) {
      Vec params_template;

      PetscCall(TaoTermCreateVecs(term, NULL, &params_template));
      PetscCall(TaoTermSetParametersTemplate(term, params_template));
      PetscCall(VecDestroy(&params_template));
    }
    PetscCall(MatSetUp(term->parameters_factory));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetFromOptions - Configure a `TaoTerm` from options

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Options Database Keys:
+ -taoterm_type <type>                              - l1, halfl2squared, `TaoTermType` for complete list
. -taoterm_solution_vec_type <type>                 - `VecType` for complete list of vector types
. -taoterm_parameters_vec_type <type>               - `VecType` for complete list of vector types
. -taoterm_parameters_mode <optional,none,required> - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`
. -taoterm_hessian_pre_is_hessian <bool>            - Whether `TaoTermCreateHessianMatricesDefault()` should make a separate preconditioning matrix
. -taoterm_hessian_mat_type <type>                  - `MatType` for Hessian matrix created by `TaoTermCreateHessianMatricesDefault()`
. -taoterm_hessian_pre_mat_type <type>              - `MatType` for Hessian preconditioning matrix created by `TaoTermCreateHessianMatricesDefault()`
. -taoterm_fd_delta <real>                          - Increment for finite difference derivative approximations in `TaoTermGradientFD()`
- -taoterm_gradient_use_fd <bool>                   - Use finite differences in `TaoTermGradient()`, overriding other user-provided or buit-in routines

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermSetFromOptions(TaoTerm term)
{
  const char *deft = TAOTERMSHELL;
  PetscBool   flg;
  char        typeName[256];
  VecType     sol_type, params_type;
  PetscBool   opt;
  PetscBool   use_fd = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (((PetscObject)term)->type_name) deft = ((PetscObject)term)->type_name;
  PetscObjectOptionsBegin((PetscObject)term);
  PetscCall(PetscOptionsFList("-taoterm_type", "TaoTerm type", "TaoTermType", TaoTermList, deft, typeName, 256, &flg));
  if (flg) {
    PetscCall(TaoTermSetType(term, typeName));
  } else {
    PetscCall(TaoTermSetType(term, deft));
  }
  PetscCall(TaoTermGetVecTypes(term, &sol_type, &params_type));
  PetscCall(PetscOptionsFList("-taoterm_solution_vec_type", "Solution vector type", "TaoTermSetVecTypes", VecList, sol_type, typeName, 256, &opt));
  if (opt) PetscCall(TaoTermSetVecTypes(term, typeName, NULL));
  PetscCall(PetscOptionsFList("-taoterm_parameters_vec_type", "Parameters vector type", "TaoTermSetVecTypes", VecList, params_type, typeName, 256, &opt));
  if (opt) PetscCall(TaoTermSetVecTypes(term, NULL, typeName));
  PetscCall(PetscOptionsEnum("-taoterm_parameters_mode", "Parameters requirement type", "TaoTermSetParametersMode", TaoTermParametersModes, (PetscEnum)term->parameters_mode, (PetscEnum *)&term->parameters_mode, NULL));
  PetscCall(PetscOptionsBool("-taoterm_hessian_pre_is_hessian", "If the Hessian and its preconditioning matrix should be the same", "TaoTermSetCreateHessianMode", term->Hpre_is_H, &term->Hpre_is_H, NULL));

  deft = MATSHELL;
  if (term->H_mattype) deft = term->H_mattype;
  PetscCall(PetscOptionsFList("-taoterm_hessian_mat_type", "Hessian mat type", "TaoTermSetCreateHessianMode", MatList, deft, typeName, 256, &opt));
  if (opt) {
    PetscCall(PetscFree(term->H_mattype));
    PetscCall(PetscStrallocpy(typeName, &term->H_mattype));
  }

  deft = MATAIJ;
  if (term->Hpre_mattype) deft = term->Hpre_mattype;
  PetscCall(PetscOptionsFList("-taoterm_hessian_pre_mat_type", "Hessian preconditioning mat type", "TaoTermSetCreateHessianMode", MatList, deft, typeName, 256, &opt));
  if (opt) {
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(PetscStrallocpy(typeName, &term->Hpre_mattype));
  }

  PetscCall(PetscOptionsBoundedReal("-taoterm_fd_delta", "Finite difference increment", "TaoTermSetFDDelta", term->fd_delta, &term->fd_delta, NULL, 0.0));
  PetscCall(PetscOptionsBool("-taoterm_gradient_use_fd", "Use finite differences in TaoTermGradient()", "TaoTermGradientUseFDPush", use_fd, &use_fd, NULL));
  if (use_fd) PetscCall(TaoTermGradientUseFDPush(term));

  PetscTryTypeMethod(term, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetType - Set the type of a `TaoTerm`

  Collective

  Input Parameter:
+ term - a `TaoTerm`
- type - a `TaoTermType`

  Options Database Key:
. -taoterm_type <type> - Sets the method; use -help for a list
   of available methods (for instance, newtonls or newtontr)

  Level: beginner

  Options Database Keys:
. -taoterm_type <type> - l1, halfl2squared, `TaoTermType` for complete list

  Note:
  New types of `TaoTerm` can be created with `TaoTermRegister()`

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreate()`,
          `TaoTermGetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermSetType(TaoTerm term, TaoTermType type)
{
  PetscErrorCode (*create)(TaoTerm);
  PetscBool issame;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);

  PetscCall(PetscObjectTypeCompare((PetscObject)term, type, &issame));
  if (issame) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoTermList, type, (void (**)(void))&create));
  PetscCheck(create, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoTerm type %s", type);

  /* Destroy the existing term information */
  PetscTryTypeMethod(term, destroy);
  term->setup_called = PETSC_FALSE;

  PetscCall((*create)(term));
  PetscCall(PetscObjectChangeTypeName((PetscObject)term, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetType - Get the type of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. type - the `TaoTermType`

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermGetType(TaoTerm term, TaoTermType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)term)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreate - Create a TaoTerm to use in `Tao` objective functions

  Collective

  Input Parameter:
. comm - communicator for MPI processes that compute the term

  Output Parameter:
. term - a new TaoTerm

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetType()`
          `TaoTermSetFromOptions()`,
          `TaoTermSetUp()`,
          `TaoTermView()`,
          `TaoTermDestroy()`,
@*/
PetscErrorCode TaoTermCreate(MPI_Comm comm, TaoTerm *term)
{
  TaoTerm     _term;
  PetscLayout zero_layout, rlayout, clayout;

  PetscFunctionBegin;
  PetscAssertPointer(term, 2);
  PetscCall(TaoInitializePackage());
  PetscCall(PetscHeaderCreate(_term, TAOTERM_CLASSID, "TaoTerm", "Objective function term", "Tao", comm, TaoTermDestroy, TaoTermView));
  PetscCall(MatCreate(comm, &_term->solution_factory));
  PetscCall(MatSetType(_term->solution_factory, MATDUMMY));
  PetscCall(MatCreate(comm, &_term->parameters_factory));
  PetscCall(MatSetType(_term->parameters_factory, MATDUMMY));
  PetscCall(PetscObjectReference((PetscObject)_term->parameters_factory));
  _term->parameters_factory_orig = _term->parameters_factory;
  PetscCall(PetscLayoutCreateFromSizes(comm, 0, 0, 1, &zero_layout));
  PetscCall(MatGetLayouts(_term->solution_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(_term->solution_factory, rlayout, zero_layout));
  PetscCall(MatGetLayouts(_term->parameters_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(_term->parameters_factory, rlayout, zero_layout));
  PetscCall(PetscLayoutDestroy(&zero_layout));
  _term->Hpre_is_H = PETSC_TRUE;
  _term->fd_delta  = 0.5 * PETSC_SQRT_MACHINE_EPSILON;
  *term            = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermObjective - Evaluate a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameter:
. value - the value of $f(x; p)$

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`,
          `TaoTermHessianMult()`,
          `TaoTermShellSetObjective()`
@*/
PetscErrorCode TaoTermObjective(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscAssertPointer(value, 4);
  if (term->ops->objective) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
  } else if (term->ops->objectiveandgradient) {
    Vec temp;

    PetscCall(PetscInfo(term, "Duplicating solution vector in order to call objective/gradient routine\n"));
    PetscCall(VecDuplicate(x, &temp));
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, temp);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscCall(VecDestroy(&temp));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have an objective function.  You should have called TaoSetObjective() or TaoTermShellSetObjective()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "TaoTerm value: %20.19e\n", (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGradient - Evaluate the gradient of a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameter:
. g - the value of $\nabla_x f(x; p)$

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`,
          `TaoTermHessianMult()`,
          `TaoTermShellSetGradient()`
@*/
PetscErrorCode TaoTermGradient(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscCheckSameComm(term, 1, g, 4);
  VecCheckSameSize(x, 2, g, 4);
  if (term->fd_grad_level > 0) {
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscCall(TaoTermGradientFD(term, x, params, g));
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  } else if (term->ops->gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  } else if (term->ops->objectiveandgradient) {
    PetscReal value;

    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, &value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
  } else SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE, "TaoTerm does not have a gradient function.  You should have called TaoSetGradient() or TaoTermShellSetGradient()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermObjectiveAndGradient - Evaluate both the value and gradient of a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameters:
+ value - the value of $f(x; p)$
- g     - the value of $\nabla_x f(x; p)$

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermHessian()`,
          `TaoTermHessianMult()`,
          `TaoTermShellSetObjectiveAndGradient()`
@*/
PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscAssertPointer(value, 4);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 5);
  PetscCheckSameComm(term, 1, g, 5);
  VecCheckSameSize(x, 2, g, 5);
  if (term->fd_grad_level > 0) {
    PetscCall(TaoTermObjective(term, x, params, value));
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscCall(TaoTermGradientFD(term, x, params, g));
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  }
  if (term->ops->objectiveandgradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objectiveandgradient, x, params, value, g);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjGradEval, term, NULL, NULL, NULL));
  } else if (term->ops->objective && term->ops->gradient) {
    PetscCall(PetscLogEventBegin(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, objective, x, params, value);
    PetscCall(PetscLogEventEnd(TAOTERM_ObjectiveEval, term, NULL, NULL, NULL));
    PetscCall(PetscLogEventBegin(TAOTERM_GradientEval, term, NULL, NULL, NULL));
    PetscUseTypeMethod(term, gradient, x, params, g);
    PetscCall(PetscLogEventEnd(TAOTERM_GradientEval, term, NULL, NULL, NULL));
  } else
    SETERRQ(PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONGSTATE,
            "TaoTerm does not have objective and gradient function.  "
            "You should have called some of the following functions: TaoSetObjective(), TaoSetGradient(), TaoSetObjectiveAndGradient(), TaoTermShellSetObjective(), TaoTermShellSetGradient(), TaoTermShellSetObjectiveAndGradient()");
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscInfo(term, "TaoTerm value: %20.19e\n", (double)(*value)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermHessian - Evaluate the Hessian of a `TaoTerm` (with respect to the solution variables) for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
- params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)

  Output Parameters:
+ H    - Hessian matrix $\nabla_x^2 f(x;p)$
- Hpre - precondiitoning matrix

  Level: beginner

  Note:
  If there is no separate preconditioning matrix, then `TaoTermHessian(term, x, params, H, NULL)`
  and `TaoTermHessian(term, x, params, H, H)` are equivalent.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessianMult()`,
          `TaoTermShellSetHessian()`
@*/
PetscErrorCode TaoTermHessian(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  if (Hpre == H) Hpre = NULL;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCall(VecLockReadPush(x));
  PetscCheckSameComm(term, 1, x, 2);
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  if (H) {
    PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
    PetscCheckSameComm(term, 1, H, 4);
  }
  if (Hpre) {
    PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
    PetscCheckSameComm(term, 1, Hpre, 5);
  }
  PetscCall(PetscLogEventBegin(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  PetscUseTypeMethod(term, hessian, x, params, H, Hpre);
  PetscCall(PetscLogEventEnd(TAOTERM_HessianEval, term, NULL, NULL, NULL));
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoTermHessianSingle - Handle the computation of optional Hessian and Hessian preconditioning matrices from a routine for computing just one.

  Logically collective

  Input Parameters:
+ term           - a `TaoTerm` context
. x              - a solution vector
. params         - (optional) a parameter vector
. H              - (optional) a matrix for the Hessian to be computed
. Hpre           - (optional) a matrix for the Hessian preconditioning matrix to be computed
. func           - a callback that computes a single copy of the Hessian matrix
- copy_structure - if `H` and `Hpre` are distinct matrices, the `str` argument to `MatCopy()` for copying `H` to `Hpre`

  Calling sequence of `func`:
+ term   - the `TaoTerm` context
. x      - the input vector
. params - the parameter vector
- H      - Hessian matrix

  Level: intermediate

  Note:
  `TaoTermHessian()` can be called with either matrix being present. All of the following are valid calling sequences\:
.vb
    TaoTermHessian(term, x, params, H, NULL);    // no preconditioning matrix requested
    TaoTermHessian(term, x, params, H, H);       // the preconditioning matrix is the Hessian matrix
    TaoTermHessian(term, x, params, H, Hpre);    // the preconditioning matrix is distinct from the Hessian matrix
    TaoTermHessian(term, x, params, NULL, Hpre); // only the preconditioning matrix is requested
.ve
  If your code does not construct the Hessian preconditioner any differently
  than the true Hessian, you can use `TaoTermHessianSingle()` in a callback
  passed to `TaoTermShellSetHessian()` to handle all of the above cases, using
  the following pattern\:
.vb
    static PetscErrorCode AppComputeHessianSingle(TaoTerm term, Vec x, Mat H)
    {
      // ... your code for computing H
    }

    static PetscErrorCode AppComputeHessian(TaoTerm term, Vec x, Mat H, Mat Hpre);
    {
      return TaoTermHessianSingle(term, x, H, Hpre, AppComputeHessianSingle, SAME_NONZERO_PATTERN);
    }

    // ... when setting the Hessian callback function, use AppComputeHessian()
    TaoTermShellSetHessian(taoterm, AppComputeHessian);
.ve

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermShellSetHessian()`
@*/
PetscErrorCode TaoTermHessianSingle(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre, PetscErrorCode (*func)(TaoTerm term, Vec x, Vec params, Mat H), MatStructure copy_structure)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  if (params) PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
  if (H) PetscValidHeaderSpecific(H, MAT_CLASSID, 4);
  if (Hpre) PetscValidHeaderSpecific(Hpre, MAT_CLASSID, 5);
  if (H) PetscCallBack("TaoTermHessiansSingle() H", (*func)(term, x, params, H));
  if (Hpre && Hpre != H) {
    if (H) PetscCall(MatCopy(H, Hpre, copy_structure));
    else PetscCallBack("TaoTermHessiansSingle() Hpre", (*func)(term, x, params, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermHessianMult - Evaluate the Hessian-vector product of a `TaoTerm` for a given set of solution variables and parameters

  Collective

  Input Parameters:
+ term   - a `TaoTerm` representing a parametric function $f(x; p)$
. x      - the solution variable $x$ in $f(x; p)$
. params - the parameters $p$ in $f(x; p)$ (may be NULL if the term is not parametric)
- v      - a vector in the solution space

  Output Parameters:
. Hv - the product Hessian matrix $\nabla_x^2 f(x;p) v$

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`
@*/
PetscErrorCode TaoTermHessianMult(TaoTerm term, Vec x, Vec params, Vec v, Vec Hv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, x, 2);
  PetscCall(VecLockReadPush(x));
  if (params) {
    PetscValidHeaderSpecific(params, VEC_CLASSID, 3);
    PetscCheckSameComm(term, 1, params, 3);
    PetscCall(VecLockReadPush(params));
  }
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  PetscCheckSameComm(term, 1, v, 4);
  PetscCall(VecLockReadPush(v));
  PetscCall(PetscLogEventBegin(TAOTERM_HessianMult, term, NULL, NULL, NULL));
  PetscUseTypeMethod(term, hessianmult, x, params, v, Hv);
  PetscCall(PetscLogEventEnd(TAOTERM_HessianMult, term, NULL, NULL, NULL));
  PetscCall(VecLockReadPop(v));
  if (params) PetscCall(VecLockReadPop(params));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveDefined - Whether this term can call `TaoTermObjective()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective is defined

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjective()`,
          `TaoTermShellSetObjective()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectivedefined) PetscUseTypeMethod(term, isobjectivedefined, is_defined);
  else *is_defined = (term->ops->objective != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsGradientDefined - Whether this term can call `TaoTermGradient()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the gradient is defined

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGradient()`,
          `TaoTermShellSetGradient()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isgradientdefined) PetscUseTypeMethod(term, isgradientdefined, is_defined);
  else *is_defined = (term->ops->gradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsObjectiveAndGradientDefined - Whether this term can call `TaoTermObjectiveAndGradient()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the objective/gradient is defined

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermShellSetObjectiveAndGradient()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->isobjectiveandgradientdefined) PetscUseTypeMethod(term, isobjectiveandgradientdefined, is_defined);
  else *is_defined = (term->ops->objectiveandgradient != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsHessianDefined - Whether this term can call `TaoTermHessian()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the hessian is defined

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermShellSetHessian()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode TaoTermIsHessianDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->ishessiandefined) PetscUseTypeMethod(term, ishessiandefined, is_defined);
  else *is_defined = (term->ops->hessian != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermIsCreateHessianMatricesDefined - Whether this term can call `TaoTermCreateHessianMatrices()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. is_defined - whether the term can create new Hessian matrices

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermShellSetCreateHessianMatrices()`,
          `TaoTermIsObjectiveDefined()`,
          `TaoTermIsGradientDefined()`,
          `TaoTermIsObjectiveAndGradientDefined()`,
          `TaoTermIsHessianDefined()`
@*/
PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm term, PetscBool *is_defined)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(is_defined, 2);
  if (term->ops->iscreatehessianmatricesdefined) PetscUseTypeMethod(term, iscreatehessianmatricesdefined, is_defined);
  else *is_defined = (term->ops->createhessianmatrices != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionSizes - Set the sizes describing the layout of the solution vector space of a `TaoTerm`.

  Logically collective

  Input Parameters:
. term - a `TaoTerm`

  Output Parameters:
+ n  - the size of a solution vector on the current MPI process (or `PETSC_DECIDE`)
. N  - the global size of a solution vector (or `PETSC_DECIDE`)
- bs - the block size of a solution vector (or `PETSC_DECIDE`)

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetSolutionSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetSolutionSizes(TaoTerm term, PetscInt n, PetscInt N, PetscInt bs)
{
  PetscLayout layout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->solution_factory, &layout, NULL));
  PetscCall(PetscLayoutSetLocalSize(layout, n));
  PetscCall(PetscLayoutSetSize(layout, N));
  PetscCall(PetscLayoutSetBlockSize(layout, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetSolutionSizes - Get the sizes describing the layout of the solution vector space of a `TaoTerm`.

  Not collective

  Input Parameters:
. term - a `TaoTerm`

  Output Parameters:
+ n  - (optional) the size of a solution vector on the current MPI process
. N  - (optional) the global size of a solution vector
- bs - (optional) the block size of a solution vector

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetSolutionSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetSolutionSizes(TaoTerm term, PetscInt *n, PetscInt *N, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (n) PetscCall(MatGetLocalSize(term->solution_factory, n, NULL));
  if (N) PetscCall(MatGetSize(term->solution_factory, N, NULL));
  if (bs) PetscCall(MatGetBlockSizes(term->solution_factory, bs, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersSizes - Set the sizes describing the layout of the parameter vector space of a `TaoTerm`.

  Logically collective

  Input Parameters:
. term - a `TaoTerm`

  Output Parameters:
+ k  - the size of a parameter vector on the current MPI process (or `PETSC_DECIDE`)
. K  - the global size of a parameter vector (or `PETSC_DECIDE`)
- bs - the block size of a parameter vector (or `PETSC_DECIDE`)

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetParametersSizes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetParametersSizes(TaoTerm term, PetscInt k, PetscInt K, PetscInt bs)
{
  PetscLayout layout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->parameters_factory, &layout, NULL));
  PetscCall(PetscLayoutSetLocalSize(layout, k));
  PetscCall(PetscLayoutSetSize(layout, K));
  PetscCall(PetscLayoutSetBlockSize(layout, bs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersSizes - Get the sizes describing the layout of the parameter vector space of a `TaoTerm`.

  Not collective

  Input Parameters:
. term - a `TaoTerm`

  Output Parameters:
+ k  - (optional) the size of a parameter vector on the current MPI process
. K  - (optional) the global size of a parameter vector
- bs - (optional) the block size of a parameter vector

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetParametersSizes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetParametersSizes(TaoTerm term, PetscInt *k, PetscInt *K, PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (k) PetscCall(MatGetLocalSize(term->parameters_factory, k, NULL));
  if (K) PetscCall(MatGetSize(term->parameters_factory, K, NULL));
  if (bs) PetscCall(MatGetBlockSizes(term->parameters_factory, bs, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetLayouts - Set the layouts describing the solution vectors and parameter vectors of a `TaoTerm`.

  Collective

  Input Parameters:
+ term              - a `TaoTerm`
. solution_layout   - the `PetscLayout` for the solution space
- parameters_layout - the `PetscLayout` for the parameter space

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetLayouts(TaoTerm term, PetscLayout solution_layout, PetscLayout parameters_layout)
{
  PetscLayout rlayout, clayout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(MatGetLayouts(term->solution_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(term->solution_factory, solution_layout, clayout));
  PetscCall(MatGetLayouts(term->parameters_factory, &rlayout, &clayout));
  PetscCall(MatSetLayouts(term->parameters_factory, parameters_layout, clayout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetLayouts - Get the layouts describing the solution vectors and parameter vectors of a `TaoTerm`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution_layout   - (optional) the `PetscLayout` for the solution space
- parameters_layout - (optional) the `PetscLayout` for the parameter space

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermSetLayouts()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetLayouts(TaoTerm term, PetscLayout *solution_layout, PetscLayout *parameters_layout)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_layout) PetscCall(MatGetLayouts(term->solution_factory, solution_layout, NULL));
  if (parameters_layout) PetscCall(MatGetLayouts(term->parameters_factory, parameters_layout, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetSolutionTemplate - Set the solution vector space to match a template vector

  Collective

  Input Parameters:
+ term         - a `TaoTerm`
- sol_template - a vector with the desired size, layout, and `VecType` of solution vectors for `TaoTerm`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm term, Vec sol_template)
{
  PetscLayout layout;
  PetscLayout clayout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(sol_template, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, sol_template, 2);
  PetscCall(VecGetType(sol_template, &vec_type));
  PetscCall(VecGetLayout(sol_template, &layout));
  PetscCall(MatGetLayouts(term->solution_factory, NULL, &clayout));
  PetscCall(MatSetLayouts(term->solution_factory, layout, clayout));
  PetscCall(MatSetVecType(term->solution_factory, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersTemplate - Set the parameter vector space to match a template vector

  Collective

  Input Parameters:
+ term            - a `TaoTerm`
- params_template - a vector with the desired size, layout, and `VecType` of parameter vectors for `TaoTerm`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetParametersTemplate(TaoTerm term, Vec params_template)
{
  PetscLayout layout;
  PetscLayout clayout;
  VecType     vec_type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(params_template, VEC_CLASSID, 2);
  PetscCheckSameComm(term, 1, params_template, 2);
  PetscCall(VecGetType(params_template, &vec_type));
  PetscCall(VecGetLayout(params_template, &layout));
  PetscCall(MatGetLayouts(term->parameters_factory, NULL, &clayout));
  PetscCall(MatSetLayouts(term->parameters_factory, layout, clayout));
  PetscCall(MatSetVecType(term->parameters_factory, vec_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetVecTypes - Set the vector types of the solution and parameter vectors of a `TaoTerm`

  Logically collective

  Input Parameters:
+ term            - a `TaoTerm`
. solution_type   - the `VecType` for the solution space
- parameters_type - the `VecType` for the parameter space

  Options Database Keys:
+ -taoterm_solution_vec_type <type>   - `VecType` for complete list of vector types
- -taoterm_parameters_vec_type <type> - `VecType` for complete list of vector types

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetLayouts()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermSetVecTypes(TaoTerm term, VecType solution_type, VecType parameters_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_type) PetscCall(MatSetVecType(term->solution_factory, solution_type));
  if (parameters_type) PetscCall(MatSetVecType(term->parameters_factory, parameters_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetVecTypes - Get the vector types of the solution and parameter vectors of a `TaoTerm`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution_type   - (optional) the `VecType` for the solution space
- parameters_type - (optional) the `VecType` for the parameter space

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermCreateVecs()`
@*/
PetscErrorCode TaoTermGetVecTypes(TaoTerm term, VecType *solution_type, VecType *parameters_type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (solution_type) PetscCall(MatGetVecType(term->solution_factory, solution_type));
  if (parameters_type) PetscCall(MatGetVecType(term->parameters_factory, parameters_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateVecs - Create a solution and/or parameter vector for a `TaoTerm`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ solution   - (optional) a compatible solution vector for `term`
- parameters - (optional) a compatible parameter vector for `term`

  Level: beginner

  Note:
  Before a `TaoTerm` can create a solution or parameter vector, you must do one of the following\:

  * Call `TaoTermSetSolutionSizes()` to describe the size and parallel layout of a solution vector
  and `TaoTermSetParametersSizes()` to describe a parameters vector (unless
  the term does not use parameters, see `TaoTermGetParametersMode()`).
  * Call `TaoTermSetLayouts()` to directly set `PetscLayout`s for the solution and/or parameters vectors.
  * Call `TaoTermSetSolutionTemplate()` and/or `TaoTermSetParametersTemplate()` to set the
  solution/parameters vector spaces to match existing `Vec`s.
  * If the `TaoTerm` is a `TAOTERMSHELL`, you can call `TaoTermShellSetCreateVecs()` to use your own code
  for creating vectors.

  You can also call `TaoTermSetVecTypes()` to set the type of vector created (e.g. `VECCUDA`).

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermShellSetCreateVecs()`,
          `TaoTermGetSolutionSizes()`,
          `TaoTermSetSolutionSizes()`,
          `TaoTermGetParametersSizes()`,
          `TaoTermSetParametersSizes()`,
          `TaoTermSetSolutionTemplate()`,
          `TaoTermSetParametersTemplate()`,
          `TaoTermGetVecTypes()`,
          `TaoTermSetVecTypes()`,
          `TaoTermGetLayouts()`,
          `TaoTermSetLayouts()`,
          `TaoTermCreateHessianMatrices()`
@*/
PetscErrorCode TaoTermCreateVecs(TaoTerm term, Vec *solution, Vec *parameters)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (term->ops->createvecs) {
    PetscUseTypeMethod(term, createvecs, solution, parameters);
  } else {
    PetscCall(MatCreateVecs(term->solution_factory, NULL, solution));
    PetscCall(MatCreateVecs(term->parameters_factory, NULL, parameters));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMatrices - Create the matrices that can be inputs to `TaoTermHessian()`

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ H    - (optional) a matrix that can store the Hessian computed in `TaoTermHessian()`
- Hpre - (optional) a preconditioner matrix that can be computed in `TaoTermHessian()`

  Level: intermediate

  Note:
  Before Hessian matrices can be created, the size of the solution vector space
  must be set (see the ways this can be done in `TaoTermCreateVecs()`).  If the
  term is a `TAOTERMSHELL`, `TaoTermShellSetCreateHessianMatrices()` must be
  called.  Most `TaoTerm`s use `TaoTermCreateHessianMatricesDefaut()` to create
  their Hessian matrices: the behavior of that function can be controlled by
  `TaoTermSetCreateHessianMode()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermShellSetCreateHessianMatrices()`,
          `TaoTermCreateVecs()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermGetCreateHessianMode()`,
          `TaoTermSetCreateHessianMode()`,
          `TaoTermIsCreateHessianDefined()`,
@*/
PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (H) PetscAssertPointer(H, 2);
  if (Hpre) PetscAssertPointer(Hpre, 3);
  PetscUseTypeMethod(term, createhessianmatrices, H, Hpre);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHessianMatricesDefault - Default routine for creating hessian matrices that can be used by many `TaoTerm` implementations

  Collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameters:
+ H    - (optional) a matrix that can store the Hessian computed in `TaoTermHessian()`
- Hpre - (optional) a preconditioner matrix that can be computed in `TaoTermHessian()`

  Level: developer

  Developer Note:
  The behavior of this routine is determined by `TaoTermSetCreateHessianMode()`.
  If `Hpre_is_H`, then the same matrix will be returned for `H` and `Hpre`,
  otherwise they will be separate matrices, with the matrix types `H_mattype` and `Hpre_mattype`.
  If either type is `MATSHELL`, then it will create a shell matrix with `TaoTermCreateHessianShell()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermGetCreateHessianMode()`,
          `TaoTermSetCreateHessianMode()`,
@*/
PetscErrorCode TaoTermCreateHessianMatricesDefault(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscBool   Hpre_is_H;
  const char *H_mattype;
  const char *Hpre_mattype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(TaoTermGetCreateHessianMode(term, &Hpre_is_H, &H_mattype, &Hpre_mattype));

  if (H || (Hpre && Hpre_is_H)) {
    Mat       _H;
    PetscBool is_shell = PETSC_FALSE;

    if (H_mattype) PetscCall(PetscStrcmp(H_mattype, MATSHELL, &is_shell));
    if (is_shell) {
      PetscCall(TaoTermCreateHessianShell(term, &_H));
    } else {
      PetscLayout sol_layout;
      VecType     sol_vec_type;

      PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &_H));
      PetscCall(TaoTermGetLayouts(term, &sol_layout, NULL));
      PetscCall(MatSetLayouts(_H, sol_layout, sol_layout));
      PetscCall(TaoTermGetVecTypes(term, &sol_vec_type, NULL));
      PetscCall(MatSetVecType(_H, sol_vec_type));
      if (H_mattype) PetscCall(MatSetType(_H, H_mattype));
      PetscCall(MatSetOption(_H, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatSetOption(_H, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
    }

    if (H) {
      PetscCall(PetscObjectReference((PetscObject)_H));
      *H = _H;
    }
    if (Hpre && Hpre_is_H) {
      PetscCall(PetscObjectReference((PetscObject)_H));
      *Hpre = _H;
    }
    PetscCall(MatDestroy(&_H));
  }
  if (Hpre && !Hpre_is_H) {
    Mat       _Hpre;
    PetscBool is_shell = PETSC_FALSE;

    if (Hpre_mattype) PetscCall(PetscStrcmp(Hpre_mattype, MATSHELL, &is_shell));
    if (is_shell) {
      PetscCall(TaoTermCreateHessianShell(term, &_Hpre));
    } else {
      PetscLayout sol_layout;
      VecType     sol_vec_type;

      PetscCall(MatCreate(PetscObjectComm((PetscObject)term), &_Hpre));
      PetscCall(TaoTermGetLayouts(term, &sol_layout, NULL));
      PetscCall(MatSetLayouts(_Hpre, sol_layout, sol_layout));
      PetscCall(TaoTermGetVecTypes(term, &sol_vec_type, NULL));
      PetscCall(MatSetVecType(_Hpre, sol_vec_type));
      if (Hpre_mattype) PetscCall(MatSetType(_Hpre, Hpre_mattype));
      PetscCall(MatSetOption(_Hpre, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatSetOption(_Hpre, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
    }
    *Hpre = _Hpre;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetCreateHessianMode - Determine the behavior of `TaoTermCreateHessianMatricesDefault()`.

  Logically collective

  Input Parameters:
+ term         - a `TaoTerm`
. Hpre_is_H    - should `TaoTermCreateHessianMatricesDefault()` make one matrix for `H` and `Hpre`?
. H_mattype    - the `MatType` to create for `H`
- Hpre_mattype - the `MatType` to create for `Hpre`

  Level: developer

  Options Database Keys:
+ -taoterm_hessian_pre_is_hessian <bool> - Whether `TaoTermCreateHessianMatrices()` should make a separate preconditioning matrix
. -taoterm_hessian_mat_type <type>       - `MatType` for Hessian matrix created by `TaoTermCreateHessianMatrices()`
- -taoterm_hessian_pre_mat_type <type>   - `MatType` for Hessian preconditioning matrix created by `TaoTermCreateHessianMatrices()`

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermGetCreateHessianMode()`,
@*/
PetscErrorCode TaoTermSetCreateHessianMode(TaoTerm term, PetscBool Hpre_is_H, const char H_mattype[], const char Hpre_mattype[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  term->Hpre_is_H = Hpre_is_H;
  if (H_mattype != term->H_mattype) {
    PetscCall(PetscFree(term->H_mattype));
    if (H_mattype) PetscCall(PetscStrallocpy(H_mattype, &term->H_mattype));
  }
  if (Hpre_mattype != term->Hpre_mattype) {
    PetscCall(PetscFree(term->Hpre_mattype));
    if (Hpre_mattype) PetscCall(PetscStrallocpy(Hpre_mattype, &term->Hpre_mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetCreateHessianMode - Get the behavior of `TaoTermCreateHessianMatricesDefault()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
+ Hpre_is_H    - (optional) should `TaoTermCreateHessianMatricesDefault()` make one matrix for `H` and `Hpre`?
. H_mattype    - (optional) the `MatType` to create for `H`
- Hpre_mattype - (optional) the `MatType` to create for `Hpre`

  Level: developer

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermHessian()`,
          `TaoTermCreateHessianMatrices()`,
          `TaoTermCreateHessianMatricesDefault()`,
          `TaoTermSetCreateHessianMode()`,
@*/
PetscErrorCode TaoTermGetCreateHessianMode(TaoTerm term, PetscBool *Hpre_is_H, const char *H_mattype[], const char *Hpre_mattype[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  if (Hpre_is_H) *Hpre_is_H = term->Hpre_is_H;
  if (H_mattype) *H_mattype = term->H_mattype;
  if (Hpre_mattype) *Hpre_mattype = term->Hpre_mattype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermDuplicate - Duplicate a `TaoTerm`

  Collective

  Input Parameters:
+ term - a `TaoTerm`
- opt  - `TAOTERM_DUPLICATE_SIZEONLY` or `TAOTERM_DUPLICATE_TYPE`

  Output Parameter:
. newterm - the duplicate `TaoTerm`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermDuplicateOption`
@*/
PetscErrorCode TaoTermDuplicate(TaoTerm term, TaoTermDuplicateOption opt, TaoTerm *newterm)
{
  VecType     solution_vec_type;
  PetscLayout rlayout, clayout;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)term), newterm));
  PetscCall(MatGetVecType(term->solution_factory, &solution_vec_type));
  PetscCall(MatGetLayouts(term->solution_factory, &rlayout, &clayout));
  PetscCall(MatSetVecType((*newterm)->solution_factory, solution_vec_type));
  PetscCall(MatSetLayouts((*newterm)->solution_factory, rlayout, clayout));
  if (opt == TAOTERM_DUPLICATE_TYPE) {
    TaoTermType type;

    PetscCall(TaoTermGetType(term, &type));
    PetscCall(TaoTermSetType(*newterm, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetParametersMode - The way a `TaoTerm` can accept parameters

  Logically collective

  Input Parameter:
+ term            - a `TaoTerm`
- parameters_mode - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

  Level: advanced

  Options Database Keys:
. -taoterm_parameters_mode <optional,none,required> - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermParametersMode`,
          `TaoTermGetParametersMode()`
@*/
PetscErrorCode TaoTermSetParametersMode(TaoTerm term, TaoTermParametersMode parameters_mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  term->parameters_mode = parameters_mode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetParametersMode - The way a `TaoTerm` can accept parameters

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. parameters_mode - `TAOTERM_PARAMETERS_OPTIONAL`, `TAOTERM_PARAMETERS_NONE`, `TAOTERM_PARAMETERS_REQUIRED`

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermParametersMode`,
          `TaoTermSetParametersMode()`
@*/
PetscErrorCode TaoTermGetParametersMode(TaoTerm term, TaoTermParametersMode *parameters_mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  *parameters_mode = term->parameters_mode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGetFDDelta - Get the increment used for finite difference derivative approximations in methods like `TaoTermGradientFD()`

  Not collective

  Input Parameter:
. term - a `TaoTerm`

  Output Parameter:
. delta - the finite difference increment

  Options Database Keys:
. -taoterm_fd_delta <delta> - the above increment

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermSetFDDelta()`,
          `TaoTermGradientFD()`,
          `TaoTermGradientUseFDPush()`,
          `TaoTermGradientUseFDPop()`,
@*/
PetscErrorCode TaoTermGetFDDelta(TaoTerm term, PetscReal *delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  *delta = term->fd_delta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermSetFDDelta - Set the increment used for finite difference derivative approximations in methods like `TaoTermGradientFD()`

  Logically collective

  Input Parameters:
+ term - a `TaoTerm`
- delta - the finite difference increment

  Options Database Keys:
. -taoterm_fd_delta <delta> - the above increment

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermGradientFD()`,
          `TaoTermGradientUseFDPush()`,
          `TaoTermGradientUseFDPop()`,
@*/
PetscErrorCode TaoTermSetFDDelta(TaoTerm term, PetscReal delta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, delta, 2);
  PetscCheck(delta > 0.0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "finite difference increment must be positive");
  term->fd_delta = delta;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGradientUseFDPush - Use finite differences instead of the user-provided or built-in gradient method in `TaoTermGradient()`.

  Logically collective

  Input Parameter:
. term - a `TaoTerm`

  Options Database Keys:
. -taoterm_gradient_use_fd <bool> - pushes 

  Level: advanced

  Note:
  This increments an internal counter: finite differences will be used whenever the counter is greater than zero.  Use
  `TaoTermGradientUseFDPop()` to undo.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermGradientUseFDPop()`,
@*/
PetscErrorCode TaoTermGradientUseFDPush(TaoTerm term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  term->fd_grad_level++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermGradientUseFDPop - Stop using finite differences in `TaoTermGradient()`.

  Logically collective

  Input Parameter:
. term - a `TaoTerm`

  Level: advanced

  Note:
  This decrements an internal counter: finite differences will be used whenever the counter is greater than zero,
  undoing `TaoTermGradientUseFDPush()`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermGetFDDelta()`,
          `TaoTermSetFDDelta()`,
          `TaoTermGradientUseFDPush()`,
@*/
PetscErrorCode TaoTermGradientUseFDPop(TaoTerm term)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  term->fd_grad_level--;
  PetscFunctionReturn(PETSC_SUCCESS);
}
