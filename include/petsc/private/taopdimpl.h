#pragma once
#include <petsc/private/taoimpl.h>
#include <petsc/private/petscimpl.h>

typedef struct _TaoPDOps *TaoPDOps;
struct _TaoPDOps {
  PetscErrorCode (*computeobjective)(TaoPD, Vec, PetscReal *, void *);
  PetscErrorCode (*computegradient)(TaoPD, Vec, Vec, void *);
  PetscErrorCode (*computeobjectiveandgradient)(TaoPD, Vec, PetscReal *, Vec, void *);
  PetscErrorCode (*setup)(TaoPD);
  PetscErrorCode (*apply)(TaoPD, Vec, PetscReal *, Vec, Vec);
  PetscErrorCode (*setfromoptions)(TaoPD, PetscOptionItems *);
  PetscErrorCode (*reset)(TaoPD);
  PetscErrorCode (*destroy)(TaoPD);
  PetscErrorCode (*applyproximalmap)(TaoPD, TaoPD, PetscReal, Vec, Vec, void *);
};

struct _p_TaoPD {
  PETSCHEADER(struct _TaoPDOps);
  void *userctx_func;
  void *userctx_grad;
  void *userctx_funcgrad;

  PetscViewer viewer;

  PetscBool setupcalled;
  PetscBool usetaoroutines;
  PetscBool usemonitor;
  PetscBool hasobjective;
  PetscBool hasgradient;
  PetscBool hasobjectiveandgradient;
  void     *data; //TODO cant tell whether I need extra pointer for ctx?

  Vec y, workvec; //TODO need to create workvec
  Vec workvec2;   //TODO only create if VM has been set? should this be in pd->data for L2?

  PetscReal scale;
  TaoPDType type;

  Tao parent_tao;
  Tao pd_tao;

  TaoPD reg_pd; // If this pd is main obj, reg_pd tracks the regularizer PD that is "attached" to itself TODO should I get bool to see if I AM an obj or not?
};

PETSC_EXTERN PetscLogEvent TAOPD_Apply;
PETSC_EXTERN PetscLogEvent TAOPD_Eval;
