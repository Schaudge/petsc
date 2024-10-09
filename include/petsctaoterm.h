#pragma once

#include <petscmat.h>
#include <petsctaotypes.h>

/* SUBMANSEC = TaoTerm */

/*S
  TaoTerm - Abstract PETSc object for a parametric real-valued function that can be a term in a `Tao` objective function.

  Level: beginner

.seealso: [](ch_tao), [](sec_tao_term),
          `TaoTermCreate()`,
          `TaoTermSetType()`,
          `TaoTermSetFromOptions()`,
          `TaoTermView()`,
          `TaoTermObjective()`,
          `TaoTermGradient()`,
          `TaoTermObjectiveAndGradient()`,
          `TaoTermHessian()`,
          `TaoTermHessianMult()`,
          `TaoTermDestroy()`,
S*/
typedef struct _p_TaoTerm *TaoTerm;

/*J
  TaoTermType - String with the name of a `TaoTerm` method

  Values:

  Level: intermediate

.seealso: [](ch_tao), [](sec_tao_term), `TaoTerm`, `TaoTermCreate()`, `TaoTermSetType()`
J*/
typedef const char *TaoTermType;

PETSC_EXTERN PetscErrorCode TaoTermRegister(const char[], PetscErrorCode (*)(TaoTerm));

PETSC_EXTERN PetscClassId      TAOTERM_CLASSID;
PETSC_EXTERN PetscFunctionList TaoTermList;

PETSC_EXTERN PetscErrorCode TaoTermCreate(MPI_Comm, TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermDestroy(TaoTerm *);
PETSC_EXTERN PetscErrorCode TaoTermView(TaoTerm, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoTermSetUp(TaoTerm);
PETSC_EXTERN PetscErrorCode TaoTermGetType(TaoTerm, TaoTermType *);
PETSC_EXTERN PetscErrorCode TaoTermSetType(TaoTerm, TaoTermType);
PETSC_EXTERN PetscErrorCode TaoTermSetFromOptions(TaoTerm);

/*E
  TaoTermParametersMode - Ways a `TaoTerm` can accept parameter vectors in `TaoTermObjective()` and related functions

  Values:
+ `TAOTERM_PARAMETERS_OPTIONAL` - the term has default parameters that will be used if parameters are omitted
. `TAOTERM_PARAMETERS_NONE`     - the term is not parametric, passing parameters is an error
- `TAOTERM_PARAMETERS_REQUIRED` - the term requires parameters, omitting parameters is an error

  Level: intermediate

.seealso: [](sec_tao_term), `TaoTerm`, `TaoTermGetParametersMode()`, `TaoTermSetParametersMode()`
E*/
typedef enum {
  TAOTERM_PARAMETERS_OPTIONAL,
  TAOTERM_PARAMETERS_NONE,
  TAOTERM_PARAMETERS_REQUIRED
} TaoTermParametersMode;
PETSC_EXTERN const char *const TaoTermParametersModes[];

PETSC_EXTERN PetscErrorCode TaoTermSetParametersMode(TaoTerm, TaoTermParametersMode);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersMode(TaoTerm, TaoTermParametersMode *);

/*E
  TaoTermDuplicateOption - Aspects to preserve when duplicating a `TaoTerm`

  Values:
+ `TAOTERM_DUPLICATE_SIZEONLY` - size of the solution space only, user must call `TaoTermSetType()`
- `TAOTERM_DUPLICATE_TYPE`     - `TaoTermType` preserved

  Level: intermediate

.seealso: `TaoTerm`, `TaoTermDuplicate()`
E*/
typedef enum {
  TAOTERM_DUPLICATE_SIZEONLY,
  TAOTERM_DUPLICATE_TYPE,
} TaoTermDuplicateOption;

PETSC_EXTERN PetscErrorCode TaoTermDuplicate(TaoTerm, TaoTermDuplicateOption, TaoTerm *);

PETSC_EXTERN PetscErrorCode TaoTermSetSolutionSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetSolutionSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetSolutionTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersSizes(TaoTerm, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TaoTermGetParametersSizes(TaoTerm, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TaoTermSetParametersTemplate(TaoTerm, Vec);
PETSC_EXTERN PetscErrorCode TaoTermSetVecTypes(TaoTerm, VecType, VecType);
PETSC_EXTERN PetscErrorCode TaoTermGetVecTypes(TaoTerm, VecType *, VecType *);
PETSC_EXTERN PetscErrorCode TaoTermSetLayouts(TaoTerm, PetscLayout, PetscLayout);
PETSC_EXTERN PetscErrorCode TaoTermGetLayouts(TaoTerm, PetscLayout *, PetscLayout *);
PETSC_EXTERN PetscErrorCode TaoTermCreateVecs(TaoTerm, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatrices(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermCreateHessianMatricesDefault(TaoTerm, Mat *, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermSetCreateHessianMode(TaoTerm, PetscBool, MatType, MatType);
PETSC_EXTERN PetscErrorCode TaoTermGetCreateHessianMode(TaoTerm, PetscBool *, MatType *, MatType *);

PETSC_EXTERN PetscErrorCode TaoTermObjective(TaoTerm, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoTermGradient(TaoTerm, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermObjectiveAndGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoTermHessian(TaoTerm, Vec, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoTermHessianSingle(TaoTerm, Vec, Vec, Mat, Mat, PetscErrorCode (*)(TaoTerm, Vec, Vec, Mat), MatStructure);
PETSC_EXTERN PetscErrorCode TaoTermHessianMult(TaoTerm, Vec, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsObjectiveAndGradientDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsHessianDefined(TaoTerm, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoTermIsCreateHessianMatricesDefined(TaoTerm, PetscBool *);

PETSC_EXTERN PetscErrorCode TaoTermCreateHessianShell(TaoTerm, Mat *);
PETSC_EXTERN PetscErrorCode TaoTermUpdateHessianShell(TaoTerm, Mat, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoTermUpdateHessianShells(TaoTerm, Vec, Vec, Mat *, Mat *);
