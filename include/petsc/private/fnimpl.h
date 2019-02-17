
#ifndef __FNIMPL_H
#define __FNIMPL_H

#include <petscfn.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool PetscFnRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscFnRegisterAll(void);

/*
  This file defines the parts of the petscfn data structure that are
  shared by all petscfn types.
*/

typedef struct _FnOps *FnOps;
struct _FnOps {
  PetscErrorCode (*createvecs)(PetscFn,Vec*,Vec*);
  PetscErrorCode (*createmats)(PetscFn,Mat*,Mat*,Mat*,Mat*,Mat*,Mat*);
  PetscErrorCode (*apply)(PetscFn,Vec,Vec);
  PetscErrorCode (*jacobianmult)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*jacobianmultadjoint)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*jacobiancreate)(PetscFn,Vec,Mat,Mat);
  PetscErrorCode (*jacobiancreateadjoint)(PetscFn,Vec,Mat,Mat);
  PetscErrorCode (*hessianmult)(PetscFn,Vec,Vec,Vec,Vec);
  PetscErrorCode (*hessiancreate)(PetscFn,Vec,Vec,Mat,Mat);
  PetscErrorCode (*scalarapply)(PetscFn,Vec,PetscScalar *);
  PetscErrorCode (*scalargradient)(PetscFn,Vec,Vec);
  PetscErrorCode (*scalarhessianmult)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*scalarhessiancreate)(PetscFn,Vec,Mat,Mat);
  PetscErrorCode (*createsubfns)(PetscFn,Vec,PetscInt,const IS[],const IS[], PetscFn *[]);
  PetscErrorCode (*destroysubfns)(PetscFn,Vec,PetscInt,const IS[],const IS[], PetscFn *[]);
  PetscErrorCode (*createsubfn)(PetscFn,Vec,PetscInt,IS,IS,PetscFn *[]);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,PetscFn);
  PetscErrorCode (*setup)(PetscFn);
  PetscErrorCode (*destroy)(PetscFn);
  PetscErrorCode (*view)(PetscFn,PetscViewer);
  PetscErrorCode (*create)(Mat);
};

#include <petscsys.h>

struct _p_PetscFn {
  PETSCHEADER(struct _FnOps);
  PetscLayout rmap,dmap;        /* range map, domain map */
  void        *data;            /* implementation-specific data */
  PetscBool   setupcalled;      /* has PetscFnSetUp() been called? */
  PetscBool   isScalar;
  VecType     rangeType, domainType;
  MatType     jacType, jacPreType;
  MatType     adjType, adjPreType;
  MatType     hesType, hesPreType;
};

#endif

