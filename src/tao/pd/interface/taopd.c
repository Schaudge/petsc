#include <petsctaopd.h> /*I "petsctaopd.h" I*/
#include <petsc/private/taopdimpl.h>

PetscFunctionList TaoPDList = NULL;

PetscClassId TAOPD_CLASSID = 0;

PetscLogEvent TAOPD_Apply;
PetscLogEvent TAOPD_Eval;

/*@C
  TaoPDViewFromOptions - View a `TaoPD` object based on values in the options database

  Collective

  Input Parameters:
+ pd   - the `TaoPD` context
. obj  - Optional object
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for available viewer options

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDView()`, `PetscObjectViewFromOptions()`, `TaoPDCreate()`
@*/
PetscErrorCode TaoPDViewFromOptions(TaoPD pd, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)pd, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDView - Prints information about the `TaoPD`

  Collective

  Input Parameters:
+ pd     - the `TaoPD` context
- viewer - visualization context

  Options Database Key:
. -tao_pd_view - Calls `TaoPDView()` at the end of each line search

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_tao), `Tao`, `TaoPD`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode TaoPDView(TaoPD pd, PetscViewer viewer)
{
  PetscBool isascii, isstring;
  TaoPDType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)pd)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(pd, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSTRING, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)pd, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    /* TODO viewer stuff */
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(TaoPDGetType(pd, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDCreate - Creates a `TaoPD` object.

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. newpd - the new `TaoPD` context

  Options Database Key:
. -tao_pd_type - select which method `Tao` should use

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDType`, `TaoPDSetType()`, `TaoPDApply()`, `TaoPDDestroy()`
@*/
PetscErrorCode TaoPDCreate(MPI_Comm comm, TaoPD *newpd)
{
  TaoPD pd;

  PetscFunctionBegin;
  PetscAssertPointer(newpd, 2);
  PetscCall(TaoPDInitializePackage());

  PetscCall(PetscHeaderCreate(pd, TAOPD_CLASSID, "TaoPD", "PD", "Tao", comm, TaoPDDestroy, TaoPDView));
  pd->scale = 1.0;
  *newpd    = pd;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDSetUp - Sets up the internal data structures for the later use
  of a `TaoPD`

  Collective

  Input Parameter:
. pd - the `TaoPD` context

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDCreate()`, `TaoPDApply()`
@*/
PetscErrorCode TaoPDSetUp(TaoPD pd)
{
  const char *default_type = TAOPDL2;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  if (pd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (!((PetscObject)pd)->type_name) PetscCall(TaoPDSetType(pd, default_type));
  PetscTryTypeMethod(pd, setup);
  if (pd->usetaoroutines) {
    PetscCall(TaoIsObjectiveDefined(pd->pd_tao, &flg));
    pd->hasobjective = flg;
    PetscCall(TaoIsGradientDefined(pd->pd_tao, &flg));
    pd->hasgradient = flg;
    PetscCall(TaoIsObjectiveAndGradientDefined(pd->pd_tao, &flg));
    pd->hasobjectiveandgradient = flg;
    /* TODO Hessian */
  } else {
    if (pd->ops->computeobjective) {
      pd->hasobjective = PETSC_TRUE;
    } else {
      pd->hasobjective = PETSC_FALSE;
    }
    if (pd->ops->computegradient) {
      pd->hasgradient = PETSC_TRUE;
    } else {
      pd->hasgradient = PETSC_FALSE;
    }
    if (pd->ops->computeobjectiveandgradient) {
      pd->hasobjectiveandgradient = PETSC_TRUE;
    } else {
      pd->hasobjectiveandgradient = PETSC_FALSE;
    }
  }
  pd->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDDestroy - Destroys the `TaoPD` context that was created with
  `TaoPDCreate()`

  Collective

  Input Parameter:
. pd - the `TaoPD` context

  Level: developer

.seealso: [](ch_ta), `TaoPDCreate()`
@*/
PetscErrorCode TaoPDDestroy(TaoPD *pd)
{
  PetscFunctionBegin;
  if (!*pd) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*pd, TAOPD_CLASSID, 1);
  if (--((PetscObject)*pd)->refct > 0) {
    *pd = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*pd)->ops->destroy) PetscCall((*(*pd)->ops->destroy)(*pd));
  if ((*pd)->usemonitor) PetscCall(PetscViewerDestroy(&(*pd)->viewer));
  PetscCall(VecDestroy(&(*pd)->workvec));
  PetscCall(VecDestroy(&(*pd)->workvec2));
  PetscCall(PetscHeaderDestroy(pd));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDSetType - Sets the algorithm used in a line search

  Collective

  Input Parameters:
+ pd   - the `TaoPD` context
- type - the `TaoPDType` selection

  Options Database Key:
. -tao_pd_type <type> - select which method Tao should use at runtime

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDType`, `TaoPDCreate()`, `TaoPDGetType()`,
@*/
PetscErrorCode TaoPDSetType(TaoPD pd, TaoPDType type)
{
  PetscErrorCode (*r)(TaoPD);
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pd, type, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(TaoPDList, type, (void (**)(void)) & r));
  PetscCheck(r, PetscObjectComm((PetscObject)pd), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested TaoPD type %s", type);
  PetscTryTypeMethod(pd, destroy);
  pd->scale = 1.0;

  pd->ops->setup          = NULL;
  pd->ops->apply          = NULL;
  pd->ops->setfromoptions = NULL;
  pd->ops->destroy        = NULL;
  pd->setupcalled         = PETSC_FALSE;
  PetscCall((*r)(pd));
  PetscCall(PetscObjectChangeTypeName((PetscObject)pd, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDSetFromOptions - Sets various `TaoPD` parameters from user
  options.

  Collective

  Input Parameter:
. pd - the `TaoPD` context

  Options Database Keys:
+ -tao_pd_type <type>   - The type of `TaoPD` (L1,L2,KL,USER)
. -tao_pd_scale <scale> - scalar scale for pd
- -tao_pd_view          - display line-search results to standard output

  Level: beginner

.seealso: `TaoPD`
@*/
PetscErrorCode TaoPDSetFromOptions(TaoPD pd)
{
  const char *default_type = TAOPDL2;
  char        type[256], monfilename[PETSC_MAX_PATH_LEN];
  PetscViewer monviewer;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)pd);
  if (((PetscObject)pd)->type_name) default_type = ((PetscObject)pd)->type_name;
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-tao_pd_type", "Tao Line Search type", "TaoPDSetType", TaoPDList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(TaoPDSetType(pd, type));
  } else if (!((PetscObject)pd)->type_name) {
    PetscCall(TaoPDSetType(pd, default_type));
  }

  PetscCall(PetscOptionsReal("-tao_pd_scale", "scale", "", pd->scale, &pd->scale, NULL));
  PetscCall(PetscOptionsString("-tao_pd_monitor", "enable the basic monitor", "TaoPDSetMonitor", "stdout", monfilename, sizeof(monfilename), &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)pd), monfilename, &monviewer));
    pd->viewer     = monviewer;
    pd->usemonitor = PETSC_TRUE;
  }
  PetscTryTypeMethod(pd, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDGetType - Gets the current pd algorithm

  Not Collective

  Input Parameter:
. pd - the `TaoPD` context

  Output Parameter:
. type - the line search algorithm in effect

  Level: developer

.seealso: `TaoPD`
@*/
PetscErrorCode TaoPDGetType(TaoPD pd, TaoPDType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)pd)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDIsUsingTaoRoutines - Checks whether the pd is using
  the standard `Tao` evaluation routines.

  Not Collective

  Input Parameter:
. pd - the `TaoPD` context

  Output Parameter:
. flg - `PETSC_TRUE` if the pd is using `Tao` evaluation routines,
        otherwise `PETSC_FALSE`

  Level: developer

.seealso: `TaoPD`
@*/
PetscErrorCode TaoPDIsUsingTaoRoutines(TaoPD pd, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  *flg = pd->usetaoroutines;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDUseTaoRoutines - Informs the `TaoPD` to use the
  objective and gradient evaluation routines from the given `Tao` object. The default.

  Logically Collective

  Input Parameters:
+ pd - the `TaoPD` context
- ts - the `Tao` context with defined objective/gradient evaluation routines

  Level: developer

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDCreate()`
@*/
PetscErrorCode TaoPDUseTaoRoutines(TaoPD pd, Tao tao)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 2);
  pd->pd_tao         = tao;
  pd->usetaoroutines = PETSC_TRUE;
  PetscCall(PetscObjectCompose((PetscObject)tao, "TaoGetParentPD", (PetscObject)pd));
  tao->is_child_pd = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ pd - the `TaoPD` context
- x  - input vector

  Output Parameter:
. f - Objective value at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDComputeGradient()`, `TaoPDComputeObjectiveAndGradient()`, `TaoPDSetObjectiveRoutine()`
@*/
PetscErrorCode TaoPDComputeObjective(TaoPD pd, Vec x, PetscReal *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheckSameComm(pd, 1, x, 2);
  PetscCall(PetscLogEventBegin(TAOPD_Eval, pd, x, NULL, NULL));
  if (pd->usetaoroutines) {
    PetscCall(TaoComputeObjective(pd->pd_tao, x, f));
  } else {
    PetscCheck(pd->ops->computeobjective || pd->ops->computeobjectiveandgradient, PetscObjectComm((PetscObject)pd), PETSC_ERR_ARG_WRONGSTATE, "PD does not have objective function set");
    if (pd->ops->computeobjective) PetscCallBack("TaoPD callback objective", (*pd->ops->computeobjective)(pd, x, f, pd->userctx_func));
    else {
      PetscCallBack("TaoPD callback objective", (*pd->ops->computeobjectiveandgradient)(pd, x, f, pd->workvec, pd->userctx_funcgrad));
    }
  }
  PetscCall(PetscLogEventEnd(TAOPD_Eval, pd, x, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ pd - the `TaoPD` context
- x  - input vector

  Output Parameters:
+ f - Objective value at `x`
- g - Gradient vector at `x`

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDComputeGradient()`, `TaoPDSetObjectiveRoutine()`
@*/
PetscErrorCode TaoPDComputeObjectiveAndGradient(TaoPD pd, Vec x, PetscReal *f, Vec g)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(pd, 1, x, 2);
  PetscCheckSameComm(pd, 1, g, 4);
  PetscCall(PetscLogEventBegin(TAOPD_Eval, pd, x, g, NULL));
  if (pd->usetaoroutines) {
    PetscCall(TaoComputeObjectiveAndGradient(pd->pd_tao, x, f, g));
  } else {
    if (pd->ops->computeobjectiveandgradient) PetscCallBack("TaoPD callback objective/gradient", (*pd->ops->computeobjectiveandgradient)(pd, x, f, g, pd->userctx_funcgrad));
    else {
      PetscCallBack("TaoPD callback objective", (*pd->ops->computeobjective)(pd, x, f, pd->userctx_func));
      PetscCallBack("TaoPD callback gradient", (*pd->ops->computegradient)(pd, x, g, pd->userctx_grad));
    }
    PetscCall(PetscInfo(pd, "TaoPD Function evaluation: %14.12e\n", (double)(*f)));
  }
  PetscCall(PetscLogEventEnd(TAOPD_Eval, pd, x, g, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ pd - the `TaoPD` context
- x  - input vector

  Output Parameter:
. g - gradient vector

  Level: developer

  Note:

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDComputeObjective()`, `TaoPDComputeObjectiveAndGradient()`, `TaoPDSetGradient()`
@*/
PetscErrorCode TaoPDComputeGradient(TaoPD pd, Vec x, Vec g)
{
  PetscReal fdummy;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheckSameComm(pd, 1, x, 2);
  PetscCheckSameComm(pd, 1, g, 3);
  if (pd->usetaoroutines) {
    PetscCall(TaoComputeGradient(pd->pd_tao, x, g));
  } else {
    PetscCall(PetscLogEventBegin(TAOPD_Eval, pd, x, g, NULL));
    if (pd->ops->computegradient) PetscCallBack("TaoPD callback gradient", (*pd->ops->computegradient)(pd, x, g, pd->userctx_grad));
    else PetscCallBack("TaoPD callback gradient", (*pd->ops->computeobjectiveandgradient)(pd, x, &fdummy, g, pd->userctx_funcgrad));
    PetscCall(PetscLogEventEnd(TAOPD_Eval, pd, x, g, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDSetCentralVector - Sets a the central vector of pd, for metric.

  Logically Collective

  Input Parameter:
+ pd - the `TaoPD` context
- y  - The initial point of the metric

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDGetCentralVector()`
@*/
PetscErrorCode TaoPDSetCentralVector(TaoPD pd, Vec y)
{
  PetscFunctionBegin;
  if (!y) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscCall(VecDestroy(&pd->y));
  pd->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDGetCentralVector - Gets a the central vector of pd, for metric.

  Logically Collective

  Input Parameter:
. pd - the `TaoPD` context

  Output Parameter:
. y - The central vector of the pd

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDSetCentralVector()`
@*/
PetscErrorCode TaoPDGetCentralVector(TaoPD pd, Vec *y)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  if (y) *y = pd->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDSetScale - Sets scale for the pd.
  If this value is not set then 1.0 is assumed.

  Logically Collective

  Input Parameters:
+ pd - the `TaoPD` context
- s  - the scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDGetStepLength()`, `TaoPDApply()`
@*/
PetscErrorCode TaoPDSetScale(TaoPD pd, PetscReal s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscValidLogicalCollectiveReal(pd, s, 2);
  pd->scale = s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoPDGetScale - Get scale of pd.

  Not Collective

  Input Parameter:
. pd - the `TaoPD` context

  Output Parameter:
. s - the current scale

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDApply()`
@*/
PetscErrorCode TaoPDGetScale(TaoPD pd, PetscReal *s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  PetscAssertPointer(s, 2);
  *s = pd->scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDRegister - Adds a pd type to the registry

  Not Collective

  Input Parameters:
+ sname - name of a new user-defined pd
- func  - routine to Create method context

  Your solver can be chosen with the procedural interface via
$    TaoPDSetType(pd, "my_pd")
  or at runtime via the option
$    -tao_pd_type my_pd

  Level: developer

  Note:
  `TaoPDRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `TaoPD`
@*/
PetscErrorCode TaoPDRegister(const char sname[], PetscErrorCode (*func)(TaoPD))
{
  PetscFunctionBegin;
  PetscCall(TaoPDInitializePackage());
  PetscCall(PetscFunctionListAdd(&TaoPDList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDAppendOptionsPrefix - Appends to the prefix used for searching
  for all `TaoPD` options in the database.

  Collective

  Input Parameters:
+ pd - the `TaoPD` solver context
- p  - the prefix string to prepend to all line search requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  This is inherited from the `Tao` object so rarely needs to be set

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDSetOptionsPrefix()`, `TaoPDGetOptionsPrefix()`
@*/
PetscErrorCode TaoPDAppendOptionsPrefix(TaoPD pd, const char p[])
{
  return PetscObjectAppendOptionsPrefix((PetscObject)pd, p);
}

/*@C
  TaoPDGetOptionsPrefix - Gets the prefix used for searching for all
  `TaoPD` options in the database

  Not Collective

  Input Parameter:
. pd - the `TaoPD` context

  Output Parameter:
. p - pointer to the prefix string used is returned

  Level: advanced

  Fortran Note:
  The user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDSetOptionsPrefix()`, `TaoPDAppendOptionsPrefix()`
@*/
PetscErrorCode TaoPDGetOptionsPrefix(TaoPD pd, const char *p[])
{
  return PetscObjectGetOptionsPrefix((PetscObject)pd, p);
}

/*@C
  TaoPDSetOptionsPrefix - Sets the prefix used for searching for all
  `TaoPD` options in the database.

  Logically Collective

  Input Parameters:
+ pd - the `TaoPD` context
- p  - the prefix string to prepend to all `pd` option requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  This is inherited from the `Tao` object so rarely needs to be set

  For example, to distinguish between the runtime options for two
  different line searches, one could call
.vb
     TaoPDSetOptionsPrefix(pd1,"sys1_")
     TaoPDSetOptionsPrefix(pd2,"sys2_")
.ve

  This would enable use of different options for each system, such as
.vb
      -sys1_tao_pd_type pdl1
      -sys2_tao_pd_type pdl2
.ve

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDAppendOptionsPrefix()`, `TaoPDGetOptionsPrefix()`
@*/
PetscErrorCode TaoPDSetOptionsPrefix(TaoPD pd, const char p[])
{
  return PetscObjectSetOptionsPrefix((PetscObject)pd, p);
}

/*@
  TaoSetRegularizer - Sets an TaoPD to Tao object.
  It treats TaoPD object as a regularizer to primary objective.
  TaoSetSolution needs to be called before this routine.

  Input Parameters:
+ tao - Tao solver context
- pd  - TaoPD context

  Level: advanced

.seealso: `TaoGetPD()`
@*/
PetscErrorCode TaoSetRegularizer(Tao tao, TaoPD pd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (pd) PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 2);
  //  PetscCall(PetscObjectReference((PetscObject)pd));
  tao->reg = pd;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)pd), PETSC_ERR_USER, "TaoSetSolution needs to be called first.");
  if (!pd->workvec) { PetscCall(VecDuplicate(tao->solution, &pd->workvec)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetPDTerm - Sets an TaoPD to Tao object.

  Input Parameters:
+ tao - Tao solver context
. pd  - TaoPD context
- i   - The number of PD

  Level: advanced

.seealso: `TaoPD`
@*/
PetscErrorCode TaoSetPDTerm(Tao tao, TaoPD pd, PetscInt i)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)pd));
  PetscCheckSameComm(tao, 1, pd, 2);
  tao->pds[i] = pd;
  if (!pd->workvec) {
    PetscCall(VecDuplicate(tao->solution, &pd->workvec));
    PetscCall(VecDuplicate(tao->solution, &pd->workvec2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetTotalNumPD - Sets total number of PDs to be added to Tao object

  Input Parameters:
+ tao - Tao solver context
- i   - The total number of PD terms

  Level: advanced

.seealso: `TaoPD`
@*/
PetscErrorCode TaoSetTotalNumPD(Tao tao, PetscInt i)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscCheck(i >= 0, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Number of PD terms cannot be negative.");
  tao->num_terms = i;
  PetscCall(PetscMalloc1(i, &tao->pds));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetPD - Gets an TaoPD that its inside Tao object.

  Input Parameters:
+ tao - Tao solver context
- i   - Index of desired TaoPD object

  Output Parameter:
. pd - TaoPD context

  Level: advanced

.seealso: `TaoSetPD()`
@*/
PetscErrorCode TaoGetPD(Tao tao, TaoPD *pd, PetscInt i)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(pd, 2);
  PetscCheck(i <= tao->num_terms - 1, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "Query index for TaoPD exceeds number of terms set");
  *pd = tao->pds[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetRegularizer - Gets an TaoPD which behaves as an regularizer to a Tao object

  Input Parameters:
. tao - Tao solver context

  Output Parameter:
. pd - TaoPD context

  Level: advanced

.seealso: `TaoSetPD()`
@*/
PetscErrorCode TaoGetRegularizer(Tao tao, TaoPD *pd)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(pd, 2);
  if (tao->is_child_pd) {
    PetscCall(PetscObjectQuery((PetscObject)tao, "TaoGetParentPD", (PetscObject *)pd));
  } else {
    *pd = tao->reg;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDSetObjective- Sets the function evaluation routine for the pd

  Logically Collective

  Input Parameters:
+ pd   - the `TaoPD` context
. func - the objective function evaluation routine
- ctx  - the (optional) user-defined context for private data

  Calling sequence of `func`:
+ pd  - the pd context
. x   - input vector
. f   - function value
- ctx - (optional) user-defined context

  Level: advanced

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDCreate()`, `TaoPDSetGradient()`, `TaoPDSetObjectiveAndGradient()`, `TaoPDUseTaoRoutines()`
@*/
PetscErrorCode TaoPDSetObjective(TaoPD pd, PetscErrorCode (*func)(TaoPD pd, Vec x, PetscReal *f, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);

  pd->ops->computeobjective = func;
  if (ctx) pd->userctx_func = ctx;
  pd->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDSetGradient - Sets the gradient evaluation routine for the pd

  Logically Collective

  Input Parameters:
+ pd   - the `TaoPD` context
. func - the gradient evaluation routine
- ctx  - the (optional) user-defined context for private data

  Calling sequence of `func`:
+ pd  - the pd object
. x   - input vector
. g   - gradient vector
- ctx - (optional) user-defined context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDCreate()`, `TaoPDSetObjective()`, `TaoPDSetObjectiveAndGradient()`, `TaoPDUseTaoRoutines()`
@*/
PetscErrorCode TaoPDSetGradient(TaoPD pd, PetscErrorCode (*func)(TaoPD pd, Vec x, Vec g, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  pd->ops->computegradient = func;
  if (ctx) pd->userctx_grad = ctx;
  pd->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDSetObjectiveAndGradient - Sets a combined objective function and gradient evaluation routine for the function to be optimized for TaoPD

  Logically Collective

  Input Parameters:
+ pd   - the `TaoPD` context
. func - the gradient function
- ctx  - [optional] user-defined context for private data for the gradient evaluation
        routine (may be `NULL`)

  Calling sequence of `func`:
+ pd - the pd object
. x   - input vector
. f   - objective value (output)
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  For some optimization methods using a combined function can be more eifficient.

.seealso: [](ch_tao), `TaoPD`, `TaoPDSetObjective()`, `TaoPDSetHessian()`, `TaoPDSetGradient()`, `TaoGetObjectiveAndGradient()`
@*/
PetscErrorCode TaoPDSetObjectiveAndGradient(TaoPD pd, PetscErrorCode (*func)(TaoPD pd, Vec x, PetscReal *f, Vec g, void *ctx), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pd, TAOPD_CLASSID, 1);
  pd->ops->computeobjectiveandgradient = func;
  if (ctx) pd->userctx_funcgrad = ctx;
  pd->usetaoroutines = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  TaoPDApplyProximalMap - Solves proximal algorithm, based on provided PD routines.
  pd0 will serve as primary objective, and pd1 will serve as regularizer.

  Collective

  Input Parameters:
+ pd0    - the `TaoPD` primary  context
. pd1    - the `TaoPD` regularizer context
. lambda - the scale of regularizer
. y      - the central vector of regularizer
. x      - the solution vector
- ctx    - pointer to context

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDCreate()`
@*/
PetscErrorCode TaoPDApplyProximalMap(TaoPD pd0, TaoPD pd1, PetscReal lambda, Vec y, Vec x, void *ctx)
{
  PetscInt low1, low2, high1, high2;

  PetscFunctionBegin;
  //TODO boilerplates
  PetscValidHeaderSpecific(pd0, TAOPD_CLASSID, 1);
  PetscValidHeaderSpecific(pd1, TAOPD_CLASSID, 2);
  PetscAssertPointer(&lambda, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscTryTypeMethod(pd0, applyproximalmap, pd1, lambda, y, x, ctx);
  PetscCheckSameComm(pd0, 1, y, 4);
  PetscCheckSameTypeAndComm(y, 4, x, 5);
  PetscCall(VecGetOwnershipRange(y, &low1, &high1));
  PetscCall(VecGetOwnershipRange(x, &low2, &high2));
  PetscCheck(low1 == low2 && high1 == high2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible vector local lengths");

  PetscCall(PetscLogEventBegin(TAOPD_Apply, pd0, pd1, y, x));
  PetscUseTypeMethod(pd0, applyproximalmap, pd1, lambda, y, x, ctx);
  PetscCall(PetscLogEventEnd(TAOPD_Apply, pd0, pd1, y, x));
  PetscCall(TaoPDViewFromOptions(pd0, NULL, "-tao_pd_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
