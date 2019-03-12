
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
  PetscErrorCode (*apply)(PetscFn,Vec,Vec);
  PetscErrorCode (*jacobianmult)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*jacobianmultadjoint)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*jacobianbuild)(PetscFn,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*jacobianbuildadjoint)(PetscFn,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*hessianmult)(PetscFn,Vec,Vec,Vec,Vec);
  PetscErrorCode (*hessianmultadjoint)(PetscFn,Vec,Vec,Vec,Vec);
  PetscErrorCode (*hessianbuild)(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*hessianbuildadjoint)(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*hessianbuildswap)(PetscFn,Vec,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*derivativescalar)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],PetscScalar*);
  PetscErrorCode (*derivativevec)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],Vec);
  PetscErrorCode (*derivativemat)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*);
  PetscErrorCode (*scalarderivativescalar)(PetscFn,Vec,PetscInt,const IS[],const Vec[],PetscScalar*);
  PetscErrorCode (*scalarderivativevec)(PetscFn,Vec,PetscInt,const IS[],const Vec[],Vec);
  PetscErrorCode (*scalarderivativemat)(PetscFn,Vec,PetscInt,const IS[],const Vec[],MatReuse,Mat*,Mat*);
  PetscErrorCode (*derivativemult)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[]);
  PetscErrorCode (*derivativebuild)(PetscFn,Vec,PetscInt,PetscInt,const IS[],const Vec[], MatReuse, Mat*, Mat*);
  PetscErrorCode (*scalarapply)(PetscFn,Vec,PetscScalar *);
  PetscErrorCode (*scalargradient)(PetscFn,Vec,Vec);
  PetscErrorCode (*scalarhessianmult)(PetscFn,Vec,Vec,Vec);
  PetscErrorCode (*scalarhessianbuild)(PetscFn,Vec,MatReuse,Mat*,Mat*);
  PetscErrorCode (*scalarderivativemult)(PetscFn,Vec,PetscInt,const IS[],const Vec[]);
  PetscErrorCode (*scalarderivativebuild)(PetscFn,Vec,PetscInt,const IS[],const Vec[], MatReuse, Mat*, Mat*);
  PetscErrorCode (*createsubfns)(PetscFn,Vec,PetscInt,const IS[],const IS[], PetscFn *[]);
  PetscErrorCode (*destroysubfns)(PetscInt,PetscFn *[]);
  PetscErrorCode (*createsubfn)(PetscFn,Vec,IS,IS,MatReuse,PetscFn *);
  PetscErrorCode (*createderivativefn)(PetscFn,PetscFnOperation,PetscInt,const Vec [],PetscFn *);
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
  PetscBool   test_jacmult;
  PetscBool   test_jacmultadj;
  PetscBool   test_hesmult;
  PetscBool   test_hesmultadj;
  PetscBool   test_scalgrad;
  PetscBool   test_scalhesmult;
  PetscBool   test_jacbuild;
  PetscBool   test_jacbuildadj;
  PetscBool   test_hesbuild;
  PetscBool   test_hesbuildadj;
  PetscBool   test_hesbuildswp;
  PetscBool   test_scalhesbuild;
  PetscBool   test_derfn;
  PetscBool   test_self_as_derfn;
};

#endif

