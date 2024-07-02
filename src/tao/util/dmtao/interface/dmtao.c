#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/
#include <petsc/private/dmimpl.h>  /*I "petscdm.h" I*/

PetscBool         DMTaoRegisterAllCalled = PETSC_FALSE;
PetscFunctionList DMTaoList              = NULL;

PetscClassId DMTAO_CLASSID = 0;

PetscLogEvent DMTAO_Eval;
PetscLogEvent DMTAO_ApplyProx;

static PetscErrorCode DMTaoDestroy(DMTao *kdm)
{
  PetscFunctionBegin;
  if (!*kdm) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*kdm), DMTAO_CLASSID, 1);
  PetscCall(VecDestroy(&(*kdm)->workvec));
  if (--((PetscObject)(*kdm))->refct > 0) {
    *kdm = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if ((*kdm)->ops->destroy) PetscCall(((*kdm)->ops->destroy)(*kdm));
  PetscCall(PetscHeaderDestroy(kdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetVM - Sets variable metric matrix to `DMTao` object

  Collective

  Input Parameters:
+ dm - the `DM` context
- vm - the variable metric matrix

  Level: beginner

.seealso: [](ch_tao), `DMTao`, `DMTaoGetVM()`
@*/
PetscErrorCode DMTaoSetVM(DM dm, Mat vm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  tdm->vm = vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetVM - Gets the variable metric matrix of `DMTao` object

  Collective

  Input Parameters:
. dm - the `DM` context

  Output Parameter:
. vm - the variable metric matrix

  Level: beginner

.seealso: [](ch_tao), `DMTao`, `DMTaoSetVM()`
@*/
PetscErrorCode DMTaoGetVM(DM dm, Mat *vm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  *vm = tdm->vm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoView - Prints information about the `DMTao` object

  Collective

  Input Parameters:
+ dm     - the `DM` context
- viewer - visualization context

  Options Database Key:
. -dmtao_view - Calls `DMTaoView()` at the end of `DMTaoApplyProximalMap()`.

  Level: beginner

  Notes:
  The available visualization contexts include
+     `PETSC_VIEWER_STDOUT_SELF` - standard output (default)
-     `PETSC_VIEWER_STDOUT_WORLD` - synchronized standard
  output where only the first processor opens
  the file.  All other processors send their
  data to the first processor to print.

.seealso: [](ch_tao), `DMTao`, `PetscViewerASCIIOpen()`
@*/
PetscErrorCode DMTaoView(DM dm, PetscViewer viewer)
{
  PetscBool isascii, isstring;
  DMTaoType type;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DMTAO_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(((PetscObject)tdm)->comm, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(tdm, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isstring));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)tdm, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscTryTypeMethod(tdm, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function evaluations=%" PetscInt_FMT "\n", tdm->nfeval));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of gradient evaluations=%" PetscInt_FMT "\n", tdm->ngeval));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of function/gradient evaluations=%" PetscInt_FMT "\n", tdm->nfgeval));
    PetscCall(PetscViewerASCIIPrintf(viewer, "total number of proximal mapping evaluations=%" PetscInt_FMT "\n", tdm->nproxeval));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else if (isstring) {
    PetscCall(DMTaoGetType(dm, &type));
    PetscCall(PetscViewerStringSPrintf(viewer, " %-3.3s", type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMTaoCreate(MPI_Comm comm, DMTao *kdm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscAssertPointer(kdm, 2);
  PetscCall(DMTaoInitializePackage());
  PetscCall(PetscHeaderCreate(tdm, DMTAO_CLASSID, "DMTao", "DMTao", "DMTao", comm, DMTaoDestroy, DMTaoView));
  tdm->lipschitz = 0.;
  tdm->sc        = 0.;
  tdm->lmap_norm = 0.;
  tdm->nfeval    = 0;
  tdm->ngeval    = 0;
  tdm->nfgeval   = 0;
  tdm->nproxeval = 0;
  tdm->lip_set   = PETSC_FALSE;
  tdm->sc_set    = PETSC_FALSE;
  tdm->lmap      = NULL;
  *kdm           = tdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  DMTaoCopy - copies the information in a `DMTao` to another `DMTao`

  Not Collective

  Input Parameters:
+ kdm  - Original `DMTao`
- nkdm - `DMTao` to receive the data, should have been created with `DMTaoCreate()`

  Level: developer

.seealso: `DMTao`, `DMTaoCreate()`, `DMTaoDestroy()`
*/
static PetscErrorCode DMTaoCopy(DMTao kdm, DMTao nkdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(kdm, DMTAO_CLASSID, 1);
  PetscValidHeaderSpecific(nkdm, DMTAO_CLASSID, 2);
  nkdm->ops->computeobjective            = kdm->ops->computeobjective;
  nkdm->ops->computegradient             = kdm->ops->computegradient;
  nkdm->ops->computeobjectiveandgradient = kdm->ops->computeobjectiveandgradient;
  nkdm->ops->setup                       = kdm->ops->setup;
  nkdm->ops->destroy                     = kdm->ops->destroy;
  nkdm->ops->view                        = kdm->ops->view;
  nkdm->ops->setfromoptions              = kdm->ops->setfromoptions;
  nkdm->ops->reset                       = kdm->ops->reset;

  nkdm->userctx_func     = kdm->userctx_func;
  nkdm->userctx_grad     = kdm->userctx_grad;
  nkdm->userctx_funcgrad = kdm->userctx_funcgrad;
  nkdm->data             = kdm->data;
  nkdm->parentdm         = kdm->parentdm;
  nkdm->lip_set          = kdm->lip_set;
  nkdm->sc               = kdm->sc;
  nkdm->lipschitz        = kdm->lipschitz;
  nkdm->sc               = kdm->sc;
  nkdm->sc_set           = kdm->sc_set;

  nkdm->lmap      = kdm->lmap;
  nkdm->lmap_norm = kdm->lmap_norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDMTao - get read-only private `DMTao` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameter:
. taodm - private `DMTao` context

  Level: developer

  Note:
  Use `DMGetDMTaoWrite()` if write access is needed. The DMTaoSetXXX API should be used wherever possible.

.seealso: `DMGetDMTaoWrite()`
@*/
PetscErrorCode DMGetDMTao(DM dm, DMTao *taodm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *taodm = (DMTao)dm->dmtao;
  if (!*taodm) {
    PetscCall(PetscInfo(dm, "Creating new DMTao\n"));
    PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dm), taodm));

    dm->dmtao          = (PetscObject)*taodm;
    (*taodm)->parentdm = dm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetDMTaoWrite - get write access to private `DMTao` context from a `DM`

  Not Collective

  Input Parameter:
. dm - `DM` to be used with `Tao`

  Output Parameter:
. tdm - private `DMTao` context

  Level: developer

.seealso: `DMGetDMTao()`
@*/
PetscErrorCode DMGetDMTaoWrite(DM dm, DMTao *tdm)
{
  DMTao sdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &sdm));
  PetscCheck(sdm->parentdm, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DMTao has a NULL parent DM");
  if (sdm->parentdm != dm) { /* Copy on write */
    DMTao oldsdm = sdm;
    PetscCall(PetscInfo(dm, "Copying DMTao due to write\n"));
    PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dm), &sdm));
    PetscCall(DMTaoCopy(oldsdm, sdm));
    PetscCall(DMTaoDestroy((DMTao *)&dm->dmtao));
    dm->dmtao     = (PetscObject)sdm;
    sdm->parentdm = dm;
  }
  *tdm = sdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMCopyDMTao - copies a `DMTao` context to a new `DM`

  Logically Collective

  Input Parameters:
+ dmsrc  - `DM` to obtain context from
- dmdest - `DM` to add context to

  Level: developer

  Note:
  The context is copied by reference. This function does not ensure that a context exists.

.seealso: `DMGetDMTao()`, `TaoSetDM()`
@*/
PetscErrorCode DMCopyDMTao(DM dmsrc, DM dmdest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmsrc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmdest, DM_CLASSID, 2);
  if (!dmdest->dmtao) PetscCall(DMTaoCreate(PetscObjectComm((PetscObject)dmdest), (DMTao *)&dmdest->dmtao));
  PetscCall(DMTaoCopy((DMTao)dmsrc->dmtao, (DMTao)dmdest->dmtao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoSetObjective - Sets the objective evaluation routine for `DMTao`

  Logically Collective

  Input Parameters:
+ dm   - the `DM` object
. func - objective evaluation function
- ctx  - [optional] context for objective function

  Calling sequence of `func`:
+ dm  - the `DM` object
. x   - input vector
. f   - function value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `DMTaoSetGradient()`, `DMTaoSetObjectiveAndGradient()`
@*/
PetscErrorCode DMTaoSetObjective(DM dm, PetscErrorCode (*func)(DM dm, Vec x, PetscReal *f, void *ctx), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (func || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (func) tdm->ops->computeobjective = func;
  if (ctx) tdm->userctx_func = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoSetGradient - Sets the gradient evaluation routine for `DMTao`

  Logically Collective

  Input Parameters:
+ dm   - the `DM` object
. func - the gradient function
- ctx  - [optional] context for gradient function

  Calling sequence of `func`:
+ dm  - the `DM` object
. x   - input vector
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: `DMTaoSetObjective()`, `DMTaoSetObjectiveAndGradient()`
@*/
PetscErrorCode DMTaoSetGradient(DM dm, PetscErrorCode (*func)(DM dm, Vec x, Vec g, void *ctx), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (func || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (func) tdm->ops->computegradient = func;
  if (ctx) tdm->userctx_grad = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoSetObjectiveAndGradient - Sets the objective and gradient evaluation routine for `DMTao`

  Logically Collective

  Input Parameters:
+ dm   - the `DM` object
. func - objective and gradient evaluation function
- ctx  - [optional] context for objective and gradient function

  Calling sequence of `func`:
+ dm  - the `DM` object
. x   - input vector
. f   - objective value (output)
. g   - gradient value (output)
- ctx - [optional] user-defined function context

  Level: advanced

.seealso: `DMTaoSetObjective()`, `DMTaoSetGradient()`
@*/
PetscErrorCode DMTaoSetObjectiveAndGradient(DM dm, PetscErrorCode (*func)(DM dm, Vec x, PetscReal *f, Vec g, void *ctx), void *ctx)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (func || ctx) PetscCall(DMGetDMTaoWrite(dm, &tdm));
  if (func) tdm->ops->computeobjectiveandgradient = func;
  if (ctx) tdm->userctx_funcgrad = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetFromOptions - Sets various `DMTao` parameters from user
  options.

  Collective

  Input Parameter:
. dm - the `DM` context

  Options Database Keys:
+ -dmtao_type <type>   - The type of `DMTao` (L1,L2,KL,Simplex,Shell,Python)
- -dmtao_view          - display line-search results to standard output

  Level: beginner

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoSetFromOptions(DM dm)
{
  const char *default_type = DMTAOL2;
  char        type[256];
  PetscBool   flg;
  DMTao       tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)dm);
  if (((PetscObject)dm)->type_name) default_type = ((PetscObject)dm)->type_name;
  /* Check for type from options */
  PetscCall(PetscOptionsFList("-dmtao_type", "DMTao type", "DMTaoSetType", DMTaoList, default_type, type, 256, &flg));
  if (flg) {
    PetscCall(DMTaoSetType(dm, type));
  } else if (!((PetscObject)dm)->type_name) {
    PetscCall(DMTaoSetType(dm, default_type));
  }
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscTryTypeMethod(tdm, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoViewFromOptions - View a `DMTao` object based on values in the options database

  Collective

  Input Parameters:
+ tdm  - the `DM` object
. obj  - Optional object
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for available viewer options

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoView()`, `PetscObjectViewFromOptions()`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoViewFromOptions(DM dm, PetscObject obj, const char name[])
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscObjectViewFromOptions((PetscObject)tdm, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetUp - Sets up the internal data structures for the later use
  of a `DMTao`

  Collective

  Input Parameter:
. dm - the `DMTao` context

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`, `DMTaoApply()`
@*/
PetscErrorCode DMTaoSetUp(DM dm)
{
  const char *default_type = DMTAOL2;
  PetscBool   flg;
  DMTao       tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  if (!((PetscObject)dm)->type_name) PetscCall(DMTaoSetType(dm, default_type));
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscTryTypeMethod(tdm, setup);
  if (tdm->usetaoroutines) {
    PetscCall(TaoIsObjectiveDefined(tdm->dm_subtao, &flg));
    tdm->hasobjective = flg;
    PetscCall(TaoIsGradientDefined(tdm->dm_subtao, &flg));
    tdm->hasgradient = flg;
    PetscCall(TaoIsObjectiveAndGradientDefined(tdm->dm_subtao, &flg));
    tdm->hasobjectiveandgradient = flg;
    /* TODO Hessian */
  } else {
    if (tdm->ops->computeobjective) {
      tdm->hasobjective = PETSC_TRUE;
    } else {
      tdm->hasobjective = PETSC_FALSE;
    }
    if (tdm->ops->computegradient) {
      tdm->hasgradient = PETSC_TRUE;
    } else {
      tdm->hasgradient = PETSC_FALSE;
    }
    if (tdm->ops->computeobjectiveandgradient) {
      tdm->hasobjectiveandgradient = PETSC_TRUE;
    } else {
      tdm->hasobjectiveandgradient = PETSC_FALSE;
    }
  }
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetType - Sets the algorithm used in a DMTao

  Collective

  Input Parameters:
+ dm   - the `DM` context
- type - the `DMTaoType` selection

  Options Database Key:
. -dmtao_type <type> - select which method DMTao should use at runtime

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoType`, `DMTaoCreate()`, `DMTaoGetType()`,
@*/
PetscErrorCode DMTaoSetType(DM dm, DMTaoType type)
{
  PetscErrorCode (*r)(DMTao);
  PetscBool flg;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)tdm, type, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(DMTaoList, type, (void (**)(void)) & r));
  PetscCheck(r, PetscObjectComm((PetscObject)tdm), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested DMTao type %s", type);
  PetscTryTypeMethod(tdm, destroy);

  tdm->nfeval    = 0;
  tdm->ngeval    = 0;
  tdm->nfgeval   = 0;
  tdm->nproxeval = 0;
  //TODO check if compute things are null?
  tdm->ops->setup            = NULL;
  tdm->ops->destroy          = NULL;
  tdm->ops->view             = NULL;
  tdm->ops->setfromoptions   = NULL;
  tdm->setupcalled           = PETSC_FALSE;
  PetscCall((*r)(tdm));
  PetscCall(PetscObjectChangeTypeName((PetscObject)tdm, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetType - Gets the current dm algorithm

  Not Collective

  Input Parameter:
. tdm - the `DM` context

  Output Parameter:
. type - the DMTao algorithm in effect

  Level: developer

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoGetType(DM tdm, DMTaoType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdm, DM_CLASSID, 1);
  *type = ((PetscObject)tdm)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoIsUsingTaoRoutines - Checks whether the DMTao is using
  the standard `Tao` evaluation routines.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. flg - `PETSC_TRUE` if the DMTao is using `Tao` evaluation routines,
        otherwise `PETSC_FALSE`

  Level: developer

.seealso: `DMTao`
@*/
PetscErrorCode DMTaoIsUsingTaoRoutines(DM dm, PetscBool *flg)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(flg, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  *flg = tdm->usetaoroutines;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoUseTaoRoutines - Informs the `DMTao` to use the
  objective and gradient evaluation routines from the given `Tao` object.

  Logically Collective

  Input Parameters:
+ dm  - the `DM` context
- tao - the `Tao` context with defined objective/gradient evaluation routines

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoUseTaoRoutines(DM dm, Tao tao)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->dm_subtao      = tao;
  tdm->usetaoroutines = PETSC_TRUE;
  tao->is_child_dm    = PETSC_TRUE;
  /* Unsetting Composing is done in taosolver.c at TaoDestroy() */
  PetscCall(PetscObjectCompose((PetscObject)tao, "TaoGetParentDM", (PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetCentralVector - Sets a the central vector of DMTao, for metric.

  Logically Collective

  Input Parameter:
+ dm - the `DM` context
- y  - The initial point of the metric

  Level: advanced

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetCentralVector()`
@*/
PetscErrorCode DMTaoSetCentralVector(DM dm, Vec y)
{
  DMTao tdm;

  PetscFunctionBegin;
  if (!y) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  PetscCall(VecDestroy(&tdm->y));
  tdm->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetCentralVector - Gets a the central vector of DMTao, for metric.

  Logically Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. y - The central vector of the DMTao

  Level: advanced

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoSetCentralVector()`
@*/
PetscErrorCode DMTaoGetCentralVector(DM dm, Vec *y)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(y, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (y) *y = tdm->y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetLipschitz - Sets Lipschitz constant of `DMTao` object.

  Logically Collective

  Input Parameters:
+ dm  - the `DM` context
- lip - the Lipschitz constant

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetLipschitz()`
@*/
PetscErrorCode DMTaoSetLipschitz(DM dm, PetscReal lip)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(dm, lip, 2);
  PetscCheck(lip > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "Lipschitz value has to be greater than zero.");
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->lipschitz = lip;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetLipschitz - Get Lipschitz constant of DMTao.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. lip - the current Lipschitz constant.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoGetLipschitz(DM dm, PetscReal *lip)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  *lip = tdm->lipschitz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoSetStrongConvexity - Sets strong convexity constant of of `DMTao` object.

  Logically Collective

  Input Parameters:
+ dm - the `DM` context
- sc - the strong convexity constant

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoGetStrongConvexity()`
@*/
PetscErrorCode DMTaoSetStrongConvexity(DM dm, PetscReal sc)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(dm, sc, 2);
  PetscCheck(sc >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "StrongConvexity value has to be non-negative.");
  PetscCall(DMGetDMTaoWrite(dm, &tdm));
  tdm->sc = sc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetStrongConvexity - Get strong convexity constant of DMTao.

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. sc - the current strong convexity constant.

  Level: intermediate

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoGetStrongConvexity(DM dm, PetscReal *sc)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  *sc = tdm->sc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoSetRegularizer - Sets an DMTao to Tao object.
  It treats DMTao object as a regularizer to primary objective.
  TaoSetSolution needs to be called before this routine.

  Input Parameters:
+ tao   - `Tao` solver context
. dm    - `DM` context
- scale - The scale of `DM` regularizer

  Level: advanced

.seealso: `TaoGetRegularizer()`
@*/
PetscErrorCode TaoSetRegularizer(Tao tao, DM dm, PetscReal scale)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidLogicalCollectiveReal(tao, scale, 3);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCheck(tdm, PetscObjectComm((PetscObject)tao), PETSC_ERR_USER, "DMTao has not been set for DM.");
  tao->reg       = dm;
  tao->reg_scale = scale;
  PetscCheck(tao->solution, PetscObjectComm((PetscObject)dm), PETSC_ERR_USER, "TaoSetSolution needs to be called first.");
  if (!tdm->workvec) { PetscCall(VecDuplicate(tao->solution, &tdm->workvec)); }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoGetRegularizer - Gets an `DM` which behaves as an regularizer to a Tao object

  Input Parameters:
. tao - `Tao` solver context

  Output Parameter:
. dm - `DM` object containt the regularizer

  Level: advanced

.seealso: `TaoSetRegularizer()`
@*/
PetscErrorCode TaoGetRegularizer(Tao tao, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscAssertPointer(dm, 2);
  if (tao->is_child_dm) {
    PetscCall(PetscObjectQuery((PetscObject)tao, "TaoGetParentDM", (PetscObject *)dm));
  } else {
    *dm = tao->reg;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeObjective - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameter:
. f - Objective value at `x`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeGradient()`, `DMTaoComputeObjectiveAndGradient()`, `DMTaoSetObjective()`
@*/
PetscErrorCode DMTaoComputeObjective(DM dm, Vec x, PetscReal *f)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, NULL, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeObjective(tdm->dm_subtao, x, f));
  } else {
    PetscCheck(tdm->ops->computeobjective || tdm->ops->computeobjectiveandgradient, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMTao does not have objective function set");
    if (tdm->ops->computeobjective) PetscCallBack("DMTao callback objective", (*tdm->ops->computeobjective)(dm, x, f, tdm->userctx_func));
    else {
      if (!tdm->workvec) { PetscCall(VecDuplicate(x, &tdm->workvec)); }
      PetscCallBack("DMTao callback objective/gradient", (*tdm->ops->computeobjectiveandgradient)(dm, x, f, tdm->workvec, tdm->userctx_funcgrad));
    }
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, dm, x, NULL, NULL));
  tdm->nfeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeObjectiveAndGradient - Computes the objective function value at a given point

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameters:
+ f - Objective value at `x`
- g - Gradient vector at `x`

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeGradient()`, `DMTaoSetObjective()`
@*/
PetscErrorCode DMTaoComputeObjectiveAndGradient(DM dm, Vec x, PetscReal *f, Vec g)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscAssertPointer(f, 3);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 4);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCheckSameComm(dm, 1, g, 4);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, g, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeObjectiveAndGradient(tdm->dm_subtao, x, f, g));
  } else {
    if (tdm->ops->computeobjectiveandgradient) PetscCallBack("DMTao callback objective/gradient", (*tdm->ops->computeobjectiveandgradient)(dm, x, f, g, tdm->userctx_funcgrad));
    else {
      PetscCallBack("DMTao callback objective", (*tdm->ops->computeobjective)(dm, x, f, tdm->userctx_func));
      PetscCallBack("DMTao callback gradient", (*tdm->ops->computegradient)(dm, x, g, tdm->userctx_grad));
    }
    PetscCall(PetscInfo(dm, "DMTao Function evaluation: %14.12e\n", (double)(*f)));
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, tdm, x, g, NULL));
  tdm->nfgeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoComputeGradient - Computes the gradient of the objective function

  Collective

  Input Parameters:
+ dm - the `DM` context
- x  - input vector

  Output Parameter:
. g - gradient vector

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoComputeObjective()`, `DMTaoComputeObjectiveAndGradient()`, `DMTaoSetGradient()`
@*/
PetscErrorCode DMTaoComputeGradient(DM dm, Vec x, Vec g)
{
  PetscReal fdummy;
  DMTao     tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(g, VEC_CLASSID, 3);
  PetscCheckSameComm(dm, 1, x, 2);
  PetscCheckSameComm(dm, 1, g, 3);
  PetscCall(DMGetDMTao(dm, &tdm));
  PetscCall(PetscLogEventBegin(DMTAO_Eval, tdm, x, g, NULL));
  if (tdm->usetaoroutines) {
    PetscCall(TaoComputeGradient(tdm->dm_subtao, x, g));
  } else {
    if (tdm->ops->computegradient) PetscCallBack("DMTao callback gradient", (*tdm->ops->computegradient)(dm, x, g, tdm->userctx_grad));
    else PetscCallBack("DMTao callback objective/gradient", (*tdm->ops->computeobjectiveandgradient)(dm, x, &fdummy, g, tdm->userctx_funcgrad));
  }
  PetscCall(PetscLogEventEnd(DMTAO_Eval, tdm, x, g, NULL));
  tdm->ngeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoIsObjectiveDefined - Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call `DMTaoComputeObjective()` or
  `DMTaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: [](ch_tao), `DMTao`, `DMTaoSetObjective()`, `DMTaoIsGradientDefined()`, `DMTaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode DMTaoIsObjectiveDefined(DM dm, PetscBool *flg)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (tdm->ops->computeobjective == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoIsGradientDefined - Checks to see if the user has
  declared an objective-only routine.  Useful for determining when
  it is appropriate to call `DMTaoComputeGradient()` or
  `DMTaoComputeGradientAndGradient()`

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: [](ch_tao), `DMTaoSetGradient()`, `DMTaoIsObjectiveDefined()`, `DMTaoIsObjectiveAndGradientDefined()`
@*/
PetscErrorCode DMTaoIsGradientDefined(DM dm, PetscBool *flg)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (tdm->ops->computegradient == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoIsObjectiveAndGradientDefined - Checks to see if the user has
  declared a joint objective/gradient routine.  Useful for determining when
  it is appropriate to call `DMTaoComputeObjective()` or
  `DMTaoComputeObjectiveAndGradient()`

  Not Collective

  Input Parameter:
. dm - the `DM` context

  Output Parameter:
. flg - `PETSC_TRUE` if function routine is set by user, `PETSC_FALSE` otherwise

  Level: developer

.seealso: [](ch_tao), `DMTaoSetObjectiveAndGradient()`, `DMTaoIsObjectiveDefined()`, `DMTaoIsGradientDefined()`
@*/
PetscErrorCode DMTaoIsObjectiveAndGradientDefined(DM dm, PetscBool *flg)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  if (tdm->ops->computeobjectiveandgradient == NULL) *flg = PETSC_FALSE;
  else *flg = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMTaoRegister - Adds a DMTao type to the registry

  Not Collective

  Input Parameters:
+ sname - name of a new user-defined `DMTao`
- func  - routine to Create method context

  Your solver can be chosen with the procedural interface via
$    DMTaoSetType(pd, "my_dmtao")
  or at runtime via the option
$    -dmtao_type my_dmtao

  Level: developer

  Note:
  `DMTaoRegister()` may be called multiple times to add several user-defined solvers.

.seealso: [](ch_tao), `Tao`, `DMTao`
@*/
PetscErrorCode DMTaoRegister(const char sname[], PetscErrorCode (*func)(DMTao))
{
  PetscFunctionBegin;
  PetscCall(DMTaoInitializePackage());
  PetscCall(PetscFunctionListAdd(&DMTaoList, sname, (void (*)(void))func));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoApplyProximalMap - Solves proximal algorithm, based on provided DMTao routines.
  dm0 will serve as primary objective, and dm1 will serve as regularizer.

  Collective

  Input Parameters:
+ dm0    - the `DM` primary  context
. dm1    - the `DM` regularizer context
. lambda - the scale of regularizer
. y      - the central vector of regularizer
- is_cj  - bool to denote conjugate or not. TRUE for conjugate, FALSE for regular

  Output Parameter:
. x - the solution vector

  Level: beginner

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoApplyProximalMap(DM dm0, DM dm1, PetscReal lambda, Vec y, Vec x, PetscBool is_cj)
{
  PetscInt low1, low2, high1, high2;
  DMTao    tdm0, tdm1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm0, DM_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscCall(DMGetDMTao(dm0, &tdm0));
  if (dm1) PetscCall(DMGetDMTao(dm1, &tdm1));
  PetscCheckSameComm(dm0, 1, y, 4);
  PetscCheckSameTypeAndComm(y, 4, x, 5);
  PetscCall(VecGetOwnershipRange(y, &low1, &high1));
  PetscCall(VecGetOwnershipRange(x, &low2, &high2));
  PetscCheck(low1 == low2 && high1 == high2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible vector local lengths");
  PetscCheck(lambda >= 0, PetscObjectComm((PetscObject)dm0), PETSC_ERR_USER, "Lambda scale cannot be negative");
  PetscCall(PetscLogEventBegin(DMTAO_ApplyProx, dm0, dm1, y, x));
  if (dm1) {
    PetscTryTypeMethod(tdm0, applyproximalmap, tdm1, lambda, y, x, is_cj);
  } else {
    PetscTryTypeMethod(tdm0, applyproximalmap, NULL, lambda, y, x, is_cj);
  }
  if (is_cj) PetscCall(VecAYPX(x, -1., y));
  PetscCall(PetscLogEventEnd(DMTAO_ApplyProx, dm0, dm1, y, x));

  /* TODO do we want view for both, or just the primary objective? */
  PetscCall(DMTaoViewFromOptions(dm0, NULL, "-dmtao_view"));
  if (dm1) PetscCall(DMTaoViewFromOptions(dm1, NULL, "-dmtao_view"));
  tdm0->nproxeval++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoGetParentDM - Gets the parent `DM` object of `DMTao` object.

  Collective

  Input Parameter:
. tdm - the `DMTao` object

  Output Parameter:
. dm - the `DM` object

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`
@*/
PetscErrorCode DMTaoGetParentDM(DMTao tdm, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tdm, DMTAO_CLASSID, 1);
  *dm = tdm->parentdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoReset - Some `DMTAO` may carry state information
  from one `DMTaoApplyProximalMap()` to the next.  This function resets this
  state information.

  Collective

  Input Parameter:
. dm - the `DM` object

  Level: developer

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoCreate()`, `DMTaoApplyProximalMap()`
@*/
PetscErrorCode DMTaoReset(DM dm)
{
  DMTao tdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetDMTao(dm, &tdm));
  tdm->nfeval    = 0;
  tdm->ngeval    = 0;
  tdm->nfgeval   = 0;
  tdm->nproxeval = 0;
  PetscTryTypeMethod(tdm, reset);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMTaoAppendOptionsPrefix - Appends to the prefix used for searching
  for all `DMTao` options in the database.

  Collective

  Input Parameters:
+ dm - the `DM` context
- p  - the prefix string to prepend to all line search requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  This is inherited from the `Tao` object so rarely needs to be set

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoSetOptionsPrefix()`, `DMTaoGetOptionsPrefix()`
@*/
PetscErrorCode DMTaoAppendOptionsPrefix(DM dm, const char p[])
{
  DMTao tdm;

  PetscCall(DMGetDMTao(dm, &tdm));
  return PetscObjectAppendOptionsPrefix((PetscObject)tdm, p);
}

/*@
  DMTaoGetOptionsPrefix - Gets the prefix used for searching for all
  `DMTao` options in the database

  Not Collective

  Input Parameter:
. tdm - the `DM` context

  Output Parameter:
. p - pointer to the prefix string used is returned

  Level: advanced

  Fortran Notes:
  The user should pass in a string 'prefix' of
  sufficient length to hold the prefix.

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoSetOptionsPrefix()`, `DMTaoAppendOptionsPrefix()`
@*/
PetscErrorCode DMTaoGetOptionsPrefix(DM dm, const char *p[])
{
  DMTao tdm;

  PetscCall(DMGetDMTao(dm, &tdm));
  return PetscObjectGetOptionsPrefix((PetscObject)tdm, p);
}

/*@
  DMTaoSetOptionsPrefix - Sets the prefix used for searching for all
  `DMTao` options in the database.

  Logically Collective

  Input Parameters:
+ dm - the `DM` context
- p  - the prefix string to prepend to all `tdm` option requests

  Level: advanced

  Notes:
  A hyphen (-) must NOT be given at the beginning of the prefix name.
  The first character of all runtime options is AUTOMATICALLY the hyphen.

  This is inherited from the `Tao` object so rarely needs to be set

  For example, to distinguish between the runtime options for two
  different line searches, one could call
.vb
      DMTaoSetOptionsPrefix(tdm1,"sys1_")
      DMTaoSetOptionsPrefix(tdm2,"sys2_")
.ve

  This would enable use of different options for each system, such as
.vb
      -sys1_dmtao_type l1
      -sys2_dmtao_type l2
.ve

.seealso: [](ch_tao), `Tao`, `DMTao`, `DMTaoAppendOptionsPrefix()`, `DMTaoGetOptionsPrefix()`
@*/
PetscErrorCode DMTaoSetOptionsPrefix(DM dm, const char p[])
{
  DMTao tdm;

  PetscCall(DMGetDMTao(dm, &tdm));
  return PetscObjectSetOptionsPrefix((PetscObject)tdm, p);
}
