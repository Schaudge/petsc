
#ifndef __FNIMPL_H
#define __FNIMPL_H

#include <petscfn.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/hashmapij.h>

PETSC_EXTERN PetscBool PetscFnRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscFnRegisterAll(void);

/*
  This file defines the parts of the petscfn data structure that are
  shared by all petscfn types.
*/

typedef struct _FnOps *FnOps;
struct _FnOps {
  PetscErrorCode (*createvecs)(PetscFn,IS,Vec*,IS,Vec*);
  PetscErrorCode (*apply)(PetscFn,Vec,Vec);
  PetscErrorCode (*derivativescalar)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],PetscScalar*);
  PetscErrorCode (*derivativevec)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],Vec);
  PetscErrorCode (*derivativemat)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*);
  PetscErrorCode (*derivativefn)(PetscFn,PetscInt,PetscInt,PetscInt,const IS[],const Vec [],PetscFn *);
  PetscErrorCode (*scalarapply)(PetscFn,Vec,PetscScalar *);
  PetscErrorCode (*scalarderivativescalar)(PetscFn,Vec,PetscInt,const IS[],const Vec[],PetscScalar*);
  PetscErrorCode (*scalarderivativevec)(PetscFn,Vec,PetscInt,const IS[],const Vec[],Vec);
  PetscErrorCode (*scalarderivativemat)(PetscFn,Vec,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*);
  PetscErrorCode (*scalarderivativefn)(PetscFn,PetscInt,PetscInt,const IS[],const Vec [],PetscFn *);
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
  PetscBool   setfromoptions;
  PetscBool   isScalar;
  PetscBool   test_derfn;
  PetscBool   test_self_as_derfn;
  PetscBool   test_scalar;
  PetscBool   test_vec;
  PetscBool   test_mat;
  PetscHMapIJ testedscalar;
  PetscHMapIJ testedvec;
  PetscHMapIJ testedmat;
};

#endif

