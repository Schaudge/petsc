#if !defined(TAOMAD_H)
#define TAOMAD_H
#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/ /*I "petscsnes.h" I*/
#include <petscdmshell.h>
#include <petscblaslapack.h>

typedef struct {
  PetscReal *f, *b, *h;
  PetscInt size, max_size;
} SimpleFilter;

typedef struct {
  PetscReal obj, val;
  PetscReal barrier;
  PetscReal yeTce, yiTci, vlTcl, vuTcu, zlTxl, zuTxu;
} Lagrangian;

/* only destroy the VECNEST objects during cleaning */
typedef struct {
  Vec X, Yi, Ye;                                     /* reduced primal-dual variables */
  Vec R;                                             /* reduced VECNEST containing X, Yi, Ye */
  PetscInt nR;                                       /* block sizing of combined vector space */
} ReducedSpaceVec;

/* only destroy the VECNEST objects during cleanup */
typedef struct {
  Vec X, Sxl, Sxu, Sc, Scl, Scu;                     /* primal variables */
  Vec Yi, Ye, Vl, Vu, Zl, Zu;                        /* dual variables */
  Vec P;                                             /* primal VECNEST containing X, Sc, Scu, Scl, Sxl, Sxu */
  Vec S;                                             /* slack VECNEST containing Sc, Scu, Scl, Sxl, Sxu */
  Vec Y;                                             /* dual VECNEST containing Yi, Ye, Vl, Vu, Zl, Vu */
  Vec Ys;                                            /* dual VECNEST containing only inequality terms */
  Vec F;                                             /* full-space VECNEST containing everything */
  Vec R;                                             /* reduced primal-dual VECNEST containg X, Yi, Ye */
  PetscInt nF, nP, nS, nY, nYs, nR;                  /* block sizing of combined vector spaces */
} FullSpaceVec;

typedef struct {
  /* problem definitions */
  PetscBool        unconstrained;
  PetscInt         Nx, Nxl, Nxu;                     /* optimization variable and bound sizing */
  PetscInt         Ne, Ni, Ncl, Ncu;                 /* equality and inequality constraint sizing */
  PetscInt         Np, Ns, Ny, Nr, Nf;               /* vector sizing of combined vector spaces */
  
  /* scalar parameters */
  PetscReal        mu, mu_r, mu_g, mu_min, mu_max;   /* log-barrier parameters */
  PetscReal        slack_init;                       /* safeguard for initial slack values */
  PetscReal        gnorm, cnorm;                     /* convergence norms */
  Lagrangian       *L, *Lprev, *Lwork;               /* scalar Lagrangian components */
  PetscReal        scale_max;                        /* maximum scaling parameter */
  PetscReal        Gscale;                           /* dynamic primal scaling factor */
  PetscBool        use_ipm;                          /* disable active-set and use IPM */
  PetscBool        use_filter;                       /* disable linesearch and use filter */
  
  /* core data structures */
  Vec              CeScale, CiScale;                 /* dynamic dual scaling */
  Mat              A, Ai, Ae;                        /* aliases for constraint jacobians */
  Vec              dFdX, Ci, Ce;                     /* aliases for base Tao vectors */
  IS               isXL, isXU, isIL, isIU;           /* index sets to extract non-infinity bounds */
  Vec              XL, XU, IL, IU, B;                /* useful intermediate vectors */
  Vec              Rsnes, Gsnes;                     /* VecNest for SNES */
  ReducedSpaceVec  *G, *Gprev;                       /* reduced gradient for SNES */
  FullSpaceVec     *Q, *Qprev, *Qwork, *D;           /* full-space iterate vectors */
  FullSpaceVec     *dLdQ, *dLdQprev, *dLdQwork;      /* full-space gradient vectors */
  FullSpaceVec     *W;                               /* full-space work vectors */

  /* filter attributes */
  TaoMADFilterType filter_type;
  SimpleFilter     *filter;                          /* filter structure */
  PetscReal        tau, tau_min;                     /* fraction-to-the-boundary ratio */
  PetscReal        suff_decr;                        /* Armijo condition parameter */
  PetscReal        alpha, alpha_min, alpha_fac;      /* step length parameters */
  PetscReal        lambda;
  Lagrangian       *Ltrial;
  FullSpaceVec     *Qtrial, *dLdQtrial;

  /* projected linesearch attributes */
  PetscReal        as_step, bound_tol, cons_tol;
  IS               fixedXB, activeXB, inactiveXB, inactiveCI;
  ReducedSpaceVec  *LB, *UB;

  /* SNES subsolver for NGMRES/ANDERSON */
  SNES             snes;
  PetscReal        rcond;
  PetscInt         restarts, msize;
} TAO_MAD;

PETSC_INTERN PetscErrorCode LagrangianCopy(Lagrangian*,Lagrangian*);
PETSC_EXTERN PetscErrorCode FullSpaceVecCreate(FullSpaceVec*);
PETSC_INTERN PetscErrorCode FullSpaceVecDuplicate(FullSpaceVec*,FullSpaceVec*);
PETSC_EXTERN PetscErrorCode FullSpaceVecDestroy(FullSpaceVec*);
PETSC_EXTERN PetscErrorCode ReducedSpaceVecCreate(ReducedSpaceVec*);
PETSC_INTERN PetscErrorCode ReducedSpaceVecDuplicate(ReducedSpaceVec*,ReducedSpaceVec*);
PETSC_EXTERN PetscErrorCode ReducedSpaceVecDestroy(ReducedSpaceVec*);

PETSC_INTERN PetscErrorCode FullSpaceVecGetNorms(FullSpaceVec*,NormType,PetscInt*,PetscReal**);
PETSC_INTERN PetscErrorCode FullSpaceVecPrintNorms(FullSpaceVec*,NormType);
PETSC_INTERN PetscErrorCode TaoMADCHeckLagrangianAndGradient(Tao,FullSpaceVec*,FullSpaceVec*);

PETSC_INTERN PetscErrorCode TaoMADComputeBarrierFunction(Tao,FullSpaceVec*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADComputeLagrangianAndGradient(Tao,FullSpaceVec*,Lagrangian*,FullSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADComputeReducedKKT(Tao,FullSpaceVec*,FullSpaceVec*,ReducedSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADEvaluateClosedFormUpdates(Tao,FullSpaceVec*,FullSpaceVec*,FullSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADEstimateActiveSet(Tao,Vec,Vec,PetscReal,Vec,PetscBool*);
PETSC_INTERN PetscErrorCode TaoMADEstimateMaxAlphas(Tao,FullSpaceVec*,FullSpaceVec*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADApplyFilterStep(Tao,FullSpaceVec*,FullSpaceVec*,Lagrangian*,FullSpaceVec*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADUpdateFilter(Tao,PetscReal,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode TaoMADUpdateBarrier(Tao,FullSpaceVec*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADCheckConvergence(Tao,FullSpaceVec*,Lagrangian*,FullSpaceVec*,PetscReal);
PETSC_INTERN PetscErrorCode TaoMADMonitor(Tao,void*);

PETSC_INTERN void TaoMADConvertReasonToSNES(TaoConvergedReason,SNESConvergedReason*);
PETSC_INTERN void TaoMADConvertReasonFromSNES(SNESConvergedReason,TaoConvergedReason*);
#endif
