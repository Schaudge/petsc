/*
  Common tools for constructing discretizations
*/
#if !defined(PETSCDT_H)
#define PETSCDT_H

#include <petscsys.h>

/*S
  PetscQuadrature - Quadrature rule for integration.

  Level: beginner

.seealso:  PetscQuadratureCreate(), PetscQuadratureDestroy()
S*/
typedef struct _p_PetscQuadrature *PetscQuadrature;

/*E
  PetscGaussLobattoLegendreCreateType - algorithm used to compute the Gauss-Lobatto-Legendre nodes and weights

  Level: intermediate

$  PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA - compute the nodes via linear algebra
$  PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON - compute the nodes by solving a nonlinear equation with Newton's method

E*/
typedef enum {PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON} PetscGaussLobattoLegendreCreateType;

PETSC_EXTERN PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussLobattoLegendreQuadrature(PetscInt,PetscGaussLobattoLegendreCreateType,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);

PETSC_EXTERN PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreIntegrate(PetscInt, PetscReal *, PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);

/*S
  PetscDTAltV - Alternating algebraic k-form calculations

  Level: developer
S*/
typedef struct _n_PetscDTAltV *PetscDTAltV;

PETSC_EXTERN PetscErrorCode PetscDTAltVCreate(PetscInt, PetscDTAltV *);
PETSC_EXTERN PetscErrorCode PetscDTAltVDestroy(PetscDTAltV *);
PETSC_EXTERN PetscErrorCode PetscDTAltVGetN(PetscDTAltV, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDTAltVGetSize(PetscDTAltV, PetscInt, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDTAltVApply(PetscDTAltV, PetscInt, const PetscReal *, const PetscReal *, PetscReal *);

PETSC_STATIC_INLINE void PetscDTEnumPermWithSign(PetscInt n, PetscInt k, PetscInt *work, PetscInt *perm, PetscBool *isOdd)
{
  PetscBool odd = PETSC_FALSE;
  PetscInt  i;
  PetscInt *w = &work[n - 2];

  PetscFunctionBeginHot;
  i = 2;
  for (i = 2; i <= n; i++) {
    *(w--) = k % i;
    k /= i;
  }
  for (i = 0; i < n; i++) perm[i] = i;
  for (i = 0; i < n - 1; i++) {
    PetscInt s = work[i];
    PetscInt swap = perm[i];

    perm[i] = perm[i + s];
    perm[i + s] = swap;
    odd ^= (!!s);
  }
  *isOdd = odd;
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void PetscDTEnumSubsetWithSign(PetscInt n, PetscInt k, PetscInt Nk, PetscInt j, PetscInt *subset, PetscBool *isOdd)
{
  PetscInt i, l;
  PetscBool odd;

  PetscFunctionBeginHot;
  odd = PETSC_FALSE;
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (j < Nminuskminus) {
      subset[l++] = i;
      Nk = Nminuskminus;
    } else {
      j -= Nminuskminus;
      odd ^= ((k - l) & 1);
      Nk = Nminusk;
    }
  }
  *isOdd = odd;
  PetscFunctionReturnVoid();
}

PETSC_STATIC_INLINE void PetscDTSubsetIndexWithSign(PetscInt n, PetscInt k, PetscInt Nk, const PetscInt *subset, PetscInt *index, PetscBool *isOdd)
{
  PetscInt  j = 0;
  PetscInt  i, l;
  PetscBool odd;

  PetscFunctionBeginHot;
  odd = PETSC_FALSE;
  for (i = 0, l = 0; i < n && l < k; i++) {
    PetscInt Nminuskminus = (Nk * (k - l)) / (n - i);
    PetscInt Nminusk = Nk - Nminuskminus;

    if (subset[l] == i) {
      l++;
      Nk = Nminuskminus;
    } else {
      j += Nminuskminus;
      odd ^= ((k - l) & 1);
      Nk = Nminusk;
    }
  }
  *index = j;
  *isOdd = odd;
  PetscFunctionReturnVoid();
}

#endif
