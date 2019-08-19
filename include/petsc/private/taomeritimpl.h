#ifndef __TAOMERIT_IMPL_H
#define __TAOMERIT_IMPL_H
#include <petsctao.h>
#include <petsc/private/petscimpl.h>
#include <petsctaomerit.h>

typedef struct _TaoMeritOps *TaoMeritOps;
struct _TaoMeritOps {
    PetscErrorCode (*setup)(TaoMerit);
    PetscErrorCode (*getvalue)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*getdirderiv)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*getvalueanddirderiv)(TaoMerit,PetscReal,PetscReal*,PetscReal*);
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

    PetscReal last_alpha;
    PetscReal last_value;
    PetscReal last_dirderiv;

    Vec Xinit;
    Vec Xtrial;
    Vec Ginit;
    Vec Gtrial;
    Vec step;

    Tao tao;
};

PETSC_EXTERN PetscLogEvent TAOMERIT_GetValue;
PETSC_EXTERN PetscLogEvent TAOMERIT_GetDirDeriv;
PETSC_EXTERN PetscLogEvent TAOMERIT_GetValueAndDirDeriv;

#endif
