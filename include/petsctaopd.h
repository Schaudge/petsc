#pragma once

/* SUBMANSEC = Tao */

/*S
   TaoPD - PETSc object that manages problem description for the `Tao` optimization solves

   Level: intermediate

.seealso: [](ch_tao), `TaoPDType`, `Tao`, `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
S*/
typedef struct _p_TaoPD *TaoPD;

#include <petsctao.h>

/*J
        TaoPDType - String with the name of a `TaoPD` method

   Values:
+   `TAOPDL1`    - L1 regularizer. \|x\|_1. Not valid for Metric where y has been set.
.   `TAOPDL2`    - L2 regularizer. \|x-y\|_2^2.  y = 0 if not set or NULL.
/   `TAOPDKL`    - KL Divergence. Only for Metric version where y vector has been set.
-   `TAOPDSHELL` - Shell version. Default.

  Options Database Key:
.  -tao_pd_type <type> - select which method Tao should use at runtime

  Level: beginner

.seealso: [](ch_tao), `Tao`, `TaoPD`, `TaoPDSetType()`, `TaoCreate()`, `TaoSetType()`
J*/
typedef const char *TaoPDType;
#define TAOPDL1      "l1"
#define TAOPDL2      "l2"
#define TAOPDKL      "kl"
#define TAOPDSIMPLEX "simplex"
#define TAOPDSHELL   "shell"

PETSC_EXTERN PetscClassId      TAOPD_CLASSID;
PETSC_EXTERN PetscFunctionList TaoPDList;

PETSC_EXTERN PetscErrorCode TaoPDCreate(MPI_Comm, TaoPD *);
PETSC_EXTERN PetscErrorCode TaoPDSetFromOptions(TaoPD);
PETSC_EXTERN PetscErrorCode TaoPDSetUp(TaoPD);
PETSC_EXTERN PetscErrorCode TaoPDDestroy(TaoPD *);
PETSC_EXTERN PetscErrorCode TaoPDMonitor(TaoPD, PetscInt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoPDView(TaoPD, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoPDViewFromOptions(TaoPD, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode TaoPDSetOptionsPrefix(TaoPD, const char prefix[]);
//PETSC_EXTERN PetscErrorCode TaoPDReset(TaoPD);
PETSC_EXTERN PetscErrorCode TaoPDAppendOptionsPrefix(TaoPD, const char[]);
PETSC_EXTERN PetscErrorCode TaoPDGetOptionsPrefix(TaoPD, const char *[]);
//PETSC_EXTERN PetscErrorCode TaoPDGetStepLength(TaoPD, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoPDGetCentralVector(TaoPD, Vec *);
PETSC_EXTERN PetscErrorCode TaoPDSetCentralVector(TaoPD, Vec);
PETSC_EXTERN PetscErrorCode TaoPDSetInitialStepLength(TaoPD, PetscReal);

PETSC_EXTERN PetscErrorCode TaoPDGetType(TaoPD, TaoPDType *);
PETSC_EXTERN PetscErrorCode TaoPDSetType(TaoPD, TaoPDType);

PETSC_EXTERN PetscErrorCode TaoPDIsUsingTaoRoutines(TaoPD, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoPDSetObjective(TaoPD, PetscErrorCode (*)(TaoPD, Vec, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TaoPDSetGradient(TaoPD, PetscErrorCode (*)(TaoPD, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoPDSetObjectiveAndGradient(TaoPD, PetscErrorCode (*)(TaoPD, Vec, PetscReal *, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoPDUseTaoRoutines(TaoPD, Tao);

PETSC_EXTERN PetscErrorCode TaoPDComputeObjective(TaoPD, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoPDComputeObjectiveAndGradient(TaoPD, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoPDComputeGradient(TaoPD, Vec, Vec);

//                                                          yvec, lb,       up,
PETSC_EXTERN PetscErrorCode TaoPDCreate_L1(MPI_Comm, TaoPD *, Vec, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoPDL1SetContext(TaoPD, PetscReal, PetscReal);
//                                                          yvec,vm
PETSC_EXTERN PetscErrorCode TaoPDCreate_L2(MPI_Comm, TaoPD *, Vec, Mat);
PETSC_EXTERN PetscErrorCode TaoPDL2SetContext(TaoPD, Mat);
//                                                               yvec,size,
PETSC_EXTERN PetscErrorCode TaoPDCreate_Simplex(MPI_Comm, TaoPD *, Vec, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoPDSimplexSetContext(TaoPD, PetscReal, PetscReal);

//                                              main   reg    lambda     y    x      ctx?
PETSC_EXTERN PetscErrorCode TaoPDApplyProximalMap(TaoPD, TaoPD, PetscReal, Vec, Vec, void *);

PETSC_EXTERN PetscErrorCode TaoPDSetScale(TaoPD, PetscReal);
PETSC_EXTERN PetscErrorCode TaoPDGetScale(TaoPD, PetscReal *);

PETSC_EXTERN PetscErrorCode TaoPDInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoPDFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TaoPDRegister(const char[], PetscErrorCode (*)(TaoPD));
PETSC_EXTERN PetscErrorCode TaoSetPD(Tao, TaoPD);
PETSC_EXTERN PetscErrorCode TaoGetPD(Tao, TaoPD *, PetscInt);
PETSC_EXTERN PetscErrorCode TaoPDSetTao(TaoPD, Tao);
PETSC_EXTERN PetscErrorCode TaoSetNumTerms(Tao, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetNumTerms(Tao, PetscInt *);

PETSC_EXTERN PetscErrorCode TaoSetPDTerm(Tao, TaoPD, PetscInt);
PETSC_EXTERN PetscErrorCode TaoSetTotalNumPD(Tao, PetscInt);

PETSC_EXTERN PetscErrorCode TaoSetRegularizer(Tao, TaoPD);
PETSC_EXTERN PetscErrorCode TaoGetRegularizer(Tao, TaoPD *);
