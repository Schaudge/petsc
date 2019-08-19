#ifndef PETSCTAOMERIT_H
#define PETSCTAOMERIT_H
#include <petsctao.h>

typedef struct _p_TaoMerit* TaoMerit;
typedef const char *TaoMeritType;
#define TAOMERITOBJECTIVE          "objective"
#define TAOMERITLAGRANGIAN         "lagrangian"
#define TAOMERITAUGLAG             "aug-lag"
#define TAOMERITLOGBARRIER         "log-barrier"

PETSC_EXTERN PetscClassId TAOMERIT_CLASSID;
PETSC_EXTERN PetscFunctionList TaoMeritList;

PETSC_EXTERN PetscErrorCode TaoMeritCreate(Tao, TaoMerit*);
PETSC_EXTERN PetscErrorCode TaoMeritSetFromOptions(TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritSetUp(TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritDestroy(TaoMerit*);
PETSC_EXTERN PetscErrorCode TaoMeritView(TaoMerit, PetscViewer);

PETSC_EXTERN PetscErrorCode TaoMeritSetType(TaoMerit, TaoMeritType);
PETSC_EXTERN PetscErrorCode TaoMeritGetType(TaoMerit, TaoMeritType*);

PETSC_EXTERN PetscErrorCode TaoMeritReset(TaoMerit, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoMeritGetValue(TaoMerit, PetscReal, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritGetDirDeriv(TaoMerit, PetscReal, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritGetValueAndDirDeriv(TaoMerit, PetscReal, PetscReal*, PetscReal*);

PETSC_EXTERN PetscErrorCode TaoMeritInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoMeritFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TaoMeritRegister(const char[], PetscErrorCode (*)(Tao, TaoMerit*));

#endif
