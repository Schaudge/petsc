/*
     Include file for the nonlinear function component of PETSc
*/
#ifndef __PETSCFN_H
#define __PETSCFN_H
#include <petscmat.h>

/*S
     PetscFn - Abstract PETSc object used to manage all nonlinear maps in PETSc

   Level: developer

  Concepts: nonlinear operator

.seealso:  PetscFnCreate(), PetscFnType, PetscFnSetType(), PetscFnDestroy()
S*/
typedef struct _p_PetscFn*           PetscFn;

/*J
    PetscFnType - String with the name of a PETSc operator type

   Level: developer

.seealso: PetscFnSetType(), PetscFn, PetscFnRegister()
J*/
typedef const char* PetscFnType;
#define PETSCFNDAG             "dag"    /* generalizes composite */
#define PETSCFNSHELL           "shell"
/* PETSCFNDM / PETSCFNDMPLEX / etc. would be defined lib libpetscdm */

/* Logging support */
PETSC_EXTERN PetscClassId PETSCFN_CLASSID;

PETSC_EXTERN PetscErrorCode PetscFnInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscFnRegister(const char[],PetscErrorCode(*)(PetscFn));

PETSC_EXTERN PetscErrorCode PetscFnCreate(MPI_Comm,PetscFn*);

PETSC_EXTERN PetscErrorCode PetscFnSetSizes(PetscFn,PetscInt,PetscInt,PetscInt,PetscInt);
PETSC_EXTERN PetscErrorCode PetscFnGetSizes(PetscFn,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode PetscFnLayoutsSetUp(PetscFn);
PETSC_EXTERN PetscErrorCode PetscFnGetLayouts(PetscFn,PetscLayout*,PetscLayout*);

PETSC_EXTERN PetscErrorCode PetscFnSetType(PetscFn,PetscFnType);
PETSC_EXTERN PetscErrorCode PetscFnGetType(PetscFn,PetscFnType*);

PETSC_EXTERN PetscErrorCode PetscFnSetOptionsPrefix(PetscFn,const char[]);
PETSC_EXTERN PetscErrorCode PetscFnAppendOptionsPrefix(PetscFn,const char[]);
PETSC_EXTERN PetscErrorCode PetscFnGetOptionsPrefix(PetscFn,const char*[]);
PETSC_EXTERN PetscErrorCode PetscFnSetFromOptions(PetscFn);

PETSC_EXTERN PetscErrorCode PetscFnSetUp(PetscFn);

PETSC_EXTERN PetscErrorCode PetscFnView(PetscFn,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode PetscFnViewFromOptions(PetscFn A,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,obj,name);}

PETSC_EXTERN PetscErrorCode PetscFnDestroy(PetscFn*);

PETSC_EXTERN PetscFunctionList PetscFnList;

typedef enum { PETSCFNOP_CREATEVECS,
               PETSCFNOP_APPLY,
               PETSCFNOP_JACOBIANMULT,
               PETSCFNOP_JACOBIANMULTADJOINT,
               PETSCFNOP_JACOBIANBUILD,
               PETSCFNOP_JACOBIANBUILDADJOINT,
               PETSCFNOP_HESSIANMULT,
               PETSCFNOP_HESSIANMULTADJOINT,
               PETSCFNOP_HESSIANBUILD,
               PETSCFNOP_HESSIANBUILDADJOINT,
               PETSCFNOP_HESSIANBUILDSWAP,
               PETSCFNOP_SCALARAPPLY,
               PETSCFNOP_SCALARGRADIENT,
               PETSCFNOP_SCALARHESSIANMULT,
               PETSCFNOP_SCALARHESSIANBUILD,
               PETSCFNOP_CREATESUBFNS,
               PETSCFNOP_DESTROYSUBFNS,
               PETSCFNOP_CREATESUBFN,
               PETSCFNOP_CREATEDERIVATIVEFN,
               PETSCFNOP_DESTROY,
               PETSCFNOP_VIEW,
               PETSCFNOP_DERIVATIVESCALAR,
               PETSCFNOP_DERIVATIVEVEC,
               PETSCFNOP_DERIVATIVEMAT,
               PETSCFNOP_SCALARDERIVATIVESCALAR,
               PETSCFNOP_SCALARDERIVATIVEVEC,
               PETSCFNOP_SCALARDERIVATIVEMAT,
             } PetscFnOperation;

PETSC_EXTERN PetscErrorCode PetscFnCreateVecs(PetscFn,IS,Vec*,IS,Vec*);

/* core, user friendly interface */
PETSC_EXTERN PetscErrorCode PetscFnApply(PetscFn,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnJacobianMult(PetscFn,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnJacobianMultAdjoint(PetscFn,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnJacobianBuild(PetscFn,Vec,MatReuse,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PetscFnJacobianBuildAdjoint(PetscFn,Vec,MatReuse,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PetscFnHessianMult(PetscFn,Vec,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnHessianBuild(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PetscFnHessianMultAdjoint(PetscFn,Vec,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnHessianBuildAdjoint(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
PETSC_EXTERN PetscErrorCode PetscFnHessianBuildSwap(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
/* generic interface allowing for index sets on the variations */
PETSC_EXTERN PetscErrorCode PetscFnDerivativeScalar(PetscFn,Vec,PetscInt,PetscInt,const IS[], const Vec[], PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscFnDerivativeVec(PetscFn,Vec,PetscInt,PetscInt,const IS[], const Vec[], Vec);
PETSC_EXTERN PetscErrorCode PetscFnDerivativeMat(PetscFn,Vec,PetscInt,PetscInt,const IS[], const Vec[], MatReuse, Mat*, Mat*);

/* core, allows an objective function to be a PetscFn.  If a PetscFn is
 * scalar, the vector routines will wrap scalar quantities in vectors of
 * length 1 and vectors in matrices */
PETSC_EXTERN PetscErrorCode PetscFnIsScalar(PetscFn, PetscBool *);

/* core, user friendly interface */
PETSC_EXTERN PetscErrorCode PetscFnScalarApply(PetscFn,Vec,PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscFnScalarGradient(PetscFn,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnScalarHessianMult(PetscFn,Vec,Vec,Vec);
PETSC_EXTERN PetscErrorCode PetscFnScalarHessianBuild(PetscFn,Vec,MatReuse,Mat*,Mat*);
/* generic interface allowing for index sets on the variations */
PETSC_EXTERN PetscErrorCode PetscFnScalarDerivativeScalar(PetscFn,Vec,PetscInt,const IS[], const Vec[], PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscFnScalarDerivativeVec(PetscFn,Vec,PetscInt,const IS[], const Vec[], Vec);
PETSC_EXTERN PetscErrorCode PetscFnScalarDerivativeMat(PetscFn,Vec,PetscInt,const IS[], const Vec[], MatReuse, Mat*, Mat*);

/* field split ideas */
PETSC_EXTERN PetscErrorCode PetscFnCreateSubFns(PetscFn,Vec,PetscInt,const IS[],const IS[],PetscFn *[]);
PETSC_EXTERN PetscErrorCode PetscFnDestroySubFns(PetscInt,PetscFn *[]);
PETSC_EXTERN PetscErrorCode PetscFnDestroyFns(PetscInt,PetscFn *[]);
PETSC_EXTERN PetscErrorCode PetscFnCreateSubFn(PetscFn,Vec,IS,IS,MatReuse,PetscFn *);


PETSC_EXTERN const char *PetscFnOperations[];

/* derivatives are functions too */
PETSC_EXTERN PetscErrorCode PetscFnCreateDerivativeFn(PetscFn,PetscFnOperation,PetscInt,const Vec [],PetscFn *);

/* Taylor tests */
PETSC_EXTERN PetscErrorCode PetscFnTestDerivativeMult(PetscFn,PetscFnOperation,Vec,Vec,Vec,PetscRandom,PetscReal,PetscReal,PetscReal*);
/* Matrix free comparisons */
PETSC_EXTERN PetscErrorCode PetscFnTestDerivativeBuild(PetscFn,PetscFnOperation,Mat,Vec,Vec,Vec,PetscRandom,PetscReal*,PetscReal*);
/* Instantiated function comparison */
PETSC_EXTERN PetscErrorCode PetscFnTestDerivativeFn(PetscFn,PetscFn,PetscFnOperation,PetscInt,const Vec [],Vec,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode PetscFnShellSetContext(PetscFn,void*);
PETSC_EXTERN PetscErrorCode PetscFnShellGetContext(PetscFn,void *);
PETSC_EXTERN PetscErrorCode PetscFnShellSetOperation(PetscFn,PetscFnOperation,void(*)(void));
PETSC_EXTERN PetscErrorCode PetscFnShellGetOperation(PetscFn,PetscFnOperation,void(**)(void));

/* utils */
PETSC_EXTERN PetscErrorCode PetscFnGetSuperVectors(PetscFn,PetscInt,PetscInt,const IS[],const Vec[],Vec,const Vec *[],Vec *);
PETSC_EXTERN PetscErrorCode PetscFnRestoreSuperVectors(PetscFn,PetscInt,PetscInt,const IS[],const Vec[],Vec,const Vec *[],Vec *);
PETSC_EXTERN PetscErrorCode PetscFnGetSuperMats(PetscFn,PetscInt,PetscInt,const IS[],MatReuse,Mat*,Mat*,MatReuse*,Mat**,Mat**);
PETSC_EXTERN PetscErrorCode PetscFnRestoreSuperMats(PetscFn,PetscInt,PetscInt,const IS[],MatReuse,Mat*,Mat*,MatReuse*,Mat**,Mat**);

/* Allow a library of common functions so that the user does not have to
 * repeat the boiler plate for them */
typedef const char* PetscFnShellType;
PETSC_EXTERN PetscFunctionList PetscFnShellList;
#define PETSCFNSIN         "sin"
#define PETSCFNNORMSQUARED "normsquared"
#define PETSCFNMAT         "mat"

PETSC_EXTERN PetscErrorCode PetscFnShellRegister(const char[],PetscErrorCode(*)(PetscFn));
PETSC_EXTERN PetscErrorCode PetscFnShellCreate(MPI_Comm,PetscFnShellType,PetscInt,PetscInt,PetscInt,PetscInt,void *,PetscFn *);

PETSC_EXTERN PetscErrorCode PetscFnCreateDAG(MPI_Comm,PetscInt,const IS[],PetscInt,const IS[],const PetscFn[],PetscFn*);
PETSC_EXTERN PetscErrorCode PetscFnDAGAddNoOp(PetscFn,Vec,const char [],PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFnDAGAddMat(PetscFn,PetscFn,PetscBool,const char [],PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFnDAGAddFn(PetscFn,PetscFn,PetscBool,const char [],PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFnDAGGetNode(PetscFn,PetscInt,PetscFn *,const char *[]);
PETSC_EXTERN PetscErrorCode PetscFnDAGSetEdge(PetscFn,PetscInt,PetscInt,VecScatter,VecScatter,Vec,Vec,Mat,PetscScalar);
PETSC_EXTERN PetscErrorCode PetscFnDAGGetEdge(PetscFn,PetscInt,PetscInt,VecScatter*,VecScatter*,Vec*,Vec*,Mat*,PetscScalar*);
PETSC_EXTERN PetscErrorCode PetscFnDAGCreateClosure(PetscFn,PetscInt,PetscFn *);
PETSC_EXTERN PetscErrorCode PetscFnDAGSplit(PetscFn,PetscInt,PetscInt,const PetscInt [],PetscBool,PetscFn *);

#endif
