#ifndef __TAOMERIT_IMPL_H
#define __TAOMERIT_IMPL_H
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsctaomerit.h>

typedef struct _TaoMeritOps *TaoMeritOps;
struct _TaoMeritOps {
    PetscErrorCode (*setup)(TaoMerit);
    PetscErrorCode (*eval)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*view)(TaoMerit,PetscViewer);
    PetscErrorCode (*setfromoptions)(PetscOptionItems*,TaoMerit);
    PetscErrorCode (*reset)(TaoMerit,Vec,Vec);
    PetscErrorCode (*destroy)(TaoMerit*);
};

struct _p_TaoMerit {
    PETSCHEADER(struct _TaoMeritOps);
    void *data;

    PetscBool setupcalled;
    PetscBool resetcalled;

    PetscReal last_eval;
    PetscReal last_alpha;

    Vec Xinit;
    Vec Xtrial;
    Vec step;

    Tao tao;
};

PETSC_EXTERN PetscLogEvent TAOMERIT_Eval;

#endif
