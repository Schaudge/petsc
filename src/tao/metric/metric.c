#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petscmath.h>
#include <../src/tao/metric/metric.h>

const char *const TaoMetricTypes[]  = {"L2", "DIAG", "KL", "USER", "TaoMetricType", "TAOMETRIC_", NULL};

/*@
   TaoMetricGetType - Retrieve the metric type for the metric tao.

   Input Parameter:
.  tao - the `Tao` context for the `TAOMETRIC` solver

   Output Parameter:
.  type - Metric type

   Level: advanced

.seealso: `Tao`, `TAOMETRIC`, `TaoMetricSetType()`, `TaoMetricType`
@*/
PetscErrorCode TaoMetricGetType(Tao tao, TaoMetricType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscValidPointer(type, 2);
  //Error if not METRIC TODO
  PetscUseMethod(tao, "TaoMetricGetType_C", (Tao, TaoMetricType *), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoMetricGetType_Private(Tao tao, TaoMetricType *type)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;

  PetscFunctionBegin;
  *type = mP->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   TaoMetricSetType - Determine the metric type for the metric tao.

   Input Parameters:
+  tao - the `Tao` context for the `TAOMETRIC` solver
-  type - Metric type

   Level: advanced

.seealso: `Tao`, `TAOMETRIC`, `TaoMetricGetType()`, `TaoMetricType`
@*/
PetscErrorCode TaoMetricSetType(Tao tao, TaoMetricType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao, TAO_CLASSID, 1);
  PetscTryMethod(tao, "TaoMetricSetType_C", (Tao, TaoMetricType), (tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode TaoMetricSetType_Private(Tao tao, TaoMetricType type)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;

  PetscFunctionBegin;
  PetscCheck(!tao->setupcalled, PetscObjectComm((PetscObject)tao), PETSC_ERR_ORDER, "TaoMetricSetType() must be called before TaoSetUp()");
  mP->type = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSolve_Metric(Tao tao)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "NO SOLVE FOR METRIC");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetUp_Metric(Tao tao)
{
//  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
//  TODO kinda dumb?
  PetscBool obj = PETSC_FALSE, objgrad = PETSC_FALSE, hess = PETSC_FALSE;
  PetscFunctionBegin;
  if (tao->ops->computeobjective != NULL) obj = PETSC_TRUE;
  if (tao->ops->computeobjectiveandgradient != NULL) objgrad = PETSC_TRUE;
  if (tao->ops->computehessian != NULL) hess = PETSC_TRUE;

  if (obj || objgrad || hess) {
    PetscCall(TaoMetricSetType(tao, TAO_METRIC_USER)); 
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoDestroy_Metric(Tao tao)
{
//  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoMetricGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoMetricSetType_C", NULL));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoSetFromOptions_Metric(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscOptionsHeadBegin(PetscOptionsObject, "Metric method, meant to supplement proximal mapping. Not meant as an independent solver. ");
  PetscCall(PetscOptionsEnum("-tao_metric_type", "Metric Type", "TaoMetricType", TaoMetricTypes, (PetscEnum)mP->type, (PetscEnum *)&mP->type, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoView_Metric(Tao tao, PetscViewer viewer)
{
  PetscBool   isascii;
//  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Metric Type: %s\n", "metric type"));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoCreate_METRIC(Tao tao)
{
  TAO_METRIC *mP;
  PetscFunctionBegin;
  tao->ops->setup          = TaoSetUp_Metric;
  tao->ops->solve          = TaoSolve_Metric;
  tao->ops->view           = TaoView_Metric;
  tao->ops->setfromoptions = TaoSetFromOptions_Metric;
  tao->ops->destroy        = TaoDestroy_Metric;

  PetscCall(PetscNew(&mP));
  tao->data = (void *)mP;

  mP->type = TAO_METRIC_L2;

  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoMetricGetType_C", TaoMetricGetType_Private));
  PetscCall(PetscObjectComposeFunction((PetscObject)tao, "TaoMetricSetType_C", TaoMetricSetType_Private));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoMetricGetContext(Tao tao, void *ctx)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  ctx  = mP->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoMetricSetContext(Tao tao, void *ctx)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  mP->ctx = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoMetricSetCentralVector(Tao tao, Vec y)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  mP->y = y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoMetricGetCentralVector(Tao tao, Vec *y)
{
  TAO_METRIC *mP = (TAO_METRIC *)tao->data;
  PetscFunctionBegin;
  *y = mP->y; 
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode TaoMetricCreate(MPI_Comm comm, Tao *tao, TaoMetricType type)
{
  PetscFunctionBegin;
  PetscCall(TaoCreate(comm, tao));
  PetscCall(TaoMetricSetType(*tao, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}
