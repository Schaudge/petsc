#if !defined(TAOMAD_H)
#define TAOMAD_H
#include <petsc/private/taoimpl.h>

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
  Vec F;                                             /* full-space VECNEST containing everything */
  Vec R;                                             /* reduced primal-dual VECNEST containg X, Yi, Ye */
  PetscInt nF, nP, nS, nY, nR;                       /* block sizing of combined vector spaces */
} FullSpaceVec;

typedef struct {
  PetscBool        unconstrained;
  PetscInt         Nx, Nxl, Nxu;                     /* optimization variable and bound sizing */
  PetscInt         Ne, Ni, Ncl, Ncu;                 /* equality and inequality constraint sizing */
  PetscInt         Np, Ns, Ny, Nr, Nf;               /* vector sizing of combined vector spaces */
  PetscInt         k, kmax;                          /* MAD history size */
  PetscInt         nupdates, nresets;                /* information counters */
  PetscReal        mu, mu_r, mu_g;                   /* log-barrier parameters */
  PetscReal        eta, beta, rcond;                 /* MAD parameters */
  PetscReal        slack_init;                       /* safeguard for initial slack values */
  PetscReal        gnorm, cnorm;                     /* convergence norms */
  Lagrangian       *L, *Lprev;                       /* scalar Lagrangian components */

  PetscReal        scale_max;                        /* maximum scaling parameter */
  PetscReal        Gscale;                           /* dynamic primal scaling factor */
  Vec              CeScale, CiScale;                 /* dynamic dual scaling */

  Mat              pre;                              /* MAD "preconditioner" matrix */
  Mat              Ai, Ae;                           /* aliases for constraint jacobians */
  Vec              dFdX, Ci, Ce;                     /* aliases for base Tao vectors */
  IS              isXL, isXU, isIL, isIU;           /* index sets to extract non-infinity bounds */
  Vec              XL, XU, IL, IU, B;                /* useful intermediate vectors */
  Vec              *QR;                              /* MAD history for reduced VECNEST primal-dual iterates */
  Vec              *GR;                              /* MAD history for reduced VECNEST first-order optimality */
  ReducedSpaceVec  *G, *Gprev;                       /* reduced-space gradient vectors */
  FullSpaceVec     *Q, *Qprev, *D;                   /* full-space iterate vectors */
  FullSpaceVec     *dLdQ, *dLdQprev;                 /* full-space gradient vectors */
  FullSpaceVec     *W;                               /* full-space work vectors */

  TaoMADFilterType filter_type;
  SimpleFilter     *filter;                          /* filter structure */
  PetscReal        tau;                              /* fraction-to-the-boundary ratio */
  PetscReal        suff_decr;                        /* Armijo condition parameter */
  PetscReal        alpha_min, alpha_fac, alpha_cut;  /* step length parameters */

  /* least-squares subproblem variables */
  TaoMADSubsolver  subsolver_type;
  Vec              gamma;

  /* KSPCG solution variables for the subproblem */
  Mat              GRmat;
  Vec              RHS;

  /* LAPACK DGELSS solution variables for the subproblem */
  PetscBLASInt     msize, nsize, nrhs, rank, lwork, info;
  PetscScalar      *GRarr, *rhs, *sigma, *work, *rwork;
  VecScatter       allgather;
  Vec              Gseq;
} TAO_MAD;

PETSC_INTERN PetscErrorCode LagrangianCopy(Lagrangian*,Lagrangian*);
PETSC_INTERN PetscErrorCode FullSpaceVecCreate(FullSpaceVec*);
PETSC_INTERN PetscErrorCode FullSpaceVecDuplicate(FullSpaceVec*,FullSpaceVec*);
PETSC_INTERN PetscErrorCode FullSpaceVecDestroy(FullSpaceVec*);
PETSC_INTERN PetscErrorCode ReducedSpaceVecCreate(ReducedSpaceVec*);
PETSC_INTERN PetscErrorCode ReducedSpaceVecDuplicate(ReducedSpaceVec*,ReducedSpaceVec*);
PETSC_INTERN PetscErrorCode ReducedSpaceVecDestroy(ReducedSpaceVec*);

PETSC_INTERN PetscErrorCode TaoMADComputeBarrierFunction(Tao,FullSpaceVec*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADComputeLagrangianAndGradient(Tao,FullSpaceVec*,Lagrangian*,FullSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADComputeReducedKKT(Tao,FullSpaceVec*,FullSpaceVec*,ReducedSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADEvaluateClosedFormUpdates(Tao,FullSpaceVec*,FullSpaceVec*,FullSpaceVec*);
PETSC_INTERN PetscErrorCode TaoMADTestFractionToBoundary(Tao,Vec,PetscReal,Vec,PetscBool*);
PETSC_INTERN PetscErrorCode TaoMADEstimateMaxAlphas(Tao,FullSpaceVec*,FullSpaceVec*,PetscReal*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADApplyFilterStep(Tao,FullSpaceVec*,FullSpaceVec*,Lagrangian*,FullSpaceVec*,PetscBool*);
PETSC_INTERN PetscErrorCode TaoMADUpdateFilter(Tao,PetscReal,PetscReal,PetscReal);
PETSC_INTERN PetscErrorCode TaoMADUpdateBarrier(Tao,FullSpaceVec*,PetscReal*);
PETSC_INTERN PetscErrorCode TaoMADCheckConvergence(Tao,Lagrangian*,FullSpaceVec*);

#endif
