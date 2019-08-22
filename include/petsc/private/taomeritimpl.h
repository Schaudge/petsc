#ifndef __TAOMERIT_IMPL_H
#define __TAOMERIT_IMPL_H
#include <petsc/private/taoimpl.h>
#include <petsctaomerit.h>

typedef struct _TaoMeritOps *TaoMeritOps;
struct _TaoMeritOps {
    PetscErrorCode (*setup)(TaoMerit);
    PetscErrorCode (*getvalue)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*getdirderiv)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*getvalueanddirderiv)(TaoMerit,PetscReal,PetscReal*,PetscReal*);
    PetscErrorCode (*view)(TaoMerit,PetscViewer);
    PetscErrorCode (*setfromoptions)(PetscOptionItems*,TaoMerit);
    PetscErrorCode (*reset)(TaoMerit, Vec*, Vec*);
    PetscErrorCode (*destroy)(TaoMerit*);

    PetscErrorCode (*userobjective)(TaoMerit,Vec,PetscReal*,void*);
    PetscErrorCode (*usergradient)(TaoMerit,Vec,Vec,void*);
    PetscErrorCode (*userobjandgrad)(TaoMerit,Vec,PetscReal*,Vec,void*);
    PetscErrorCode (*userhessian)(TaoMerit,Vec,Mat,Mat,void*);
    PetscErrorCode (*usercnstreq)(TaoMerit,Vec,Vec,void*);
    PetscErrorCode (*usercnstrineq)(TaoMerit,Vec,Vec,void*);
    PetscErrorCode (*userjaceq)(TaoMerit,Vec,Mat,void*);
    PetscErrorCode (*userjacineq)(TaoMerit,Vec,Mat,void*);
};

struct _p_TaoMerit {
    PETSCHEADER(struct _TaoMeritOps);
    void *data;
    void *user_obj;
    void *user_grad;
    void *user_objgrad;
    void *user_hess;
    void *user_cnstreq;
    void *user_cnstrineq;
    void *user_jaceq;
    void *user_jacineq;

    PetscBool setupcalled;
    PetscBool resetcalled;
    PetscBool bounded;
    PetscBool use_tao;

    PetscReal last_alpha;
    PetscReal last_value;
    PetscReal last_dirderiv;

    Mat H, Hpre;
    Mat Jeq, Jeq_pre;
    Mat Jineq, Jineq_pre;

    Vec Xinit;
    Vec Ceq;
    Vec Cineq;
    Vec step;

    Vec Xtrial;
    Vec Gtrial;

    Tao tao;
};

PETSC_EXTERN PetscLogEvent TAOMERIT_GetValue;
PETSC_EXTERN PetscLogEvent TAOMERIT_GetDirDeriv;
PETSC_EXTERN PetscLogEvent TAOMERIT_GetValueAndDirDeriv;

PETSC_INTERN PetscErrorCode TaoMeritComputeObjective(TaoMerit, Vec, PetscReal*);
PETSC_INTERN PetscErrorCode TaoMeritComputeGradient(TaoMerit, Vec, Vec);
PETSC_INTERN PetscErrorCode TaoMeritComputeObjectiveAndGradient(TaoMerit, Vec, PetscReal*, Vec);
PETSC_INTERN PetscErrorCode TaoMeritComputeHessian(TaoMerit, Vec, Mat, Mat);
PETSC_INTERN PetscErrorCode TaoMeritComputeEqualityConstraints(TaoMerit, Vec, Vec);
PETSC_INTERN PetscErrorCode TaoMeritComputeInequalityConstraints(TaoMerit, Vec, Vec);
PETSC_INTERN PetscErrorCode TaoMeritComputeEqualityJacobian(TaoMerit, Vec, Mat);
PETSC_INTERN PetscErrorCode TaoMeritComputeInequalityJacobian(TaoMerit, Vec, Mat);

#endif
