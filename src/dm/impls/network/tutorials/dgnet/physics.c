/*
    Currently just a dumping ground for physics functions needed for the various tests. Namely flux functions
    eigenvalues, characteristic decompositions, initial condition specifications, exact solutions,
    network riemann solvers (to be removed as class is built for them specifically)
*/

#include "physics.h"
#include "petscerror.h"
#include "petscmat.h"
#include "petscsystypes.h"
/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
  PetscReal parenth;
  PetscReal parentv;
} ShallowCtx;

static inline PetscErrorCode ShallowFlux(void *ctx, const PetscReal *u, PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx *)ctx;

  PetscFunctionBeginUser;
  f[0] = u[1];
  f[1] = PetscSqr(u[1]) / u[0] + 0.5 * phys->gravity * PetscSqr(u[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void ShallowFluxVoid(void *ctx, const PetscReal *u, PetscReal *f)
{
  ShallowCtx *phys = (ShallowCtx *)ctx;
  f[0]             = u[1];
  f[1]             = PetscSqr(u[1]) / u[0] + 0.5 * phys->gravity * PetscSqr(u[0]);
}

static inline void ShallowEig(void *ctx, const PetscReal *u, PetscReal *eig)
{
  ShallowCtx *phys = (ShallowCtx *)ctx;
  eig[0]           = u[1] / u[0] - PetscSqrtReal(phys->gravity * u[0]); /*left wave*/
  eig[1]           = u[1] / u[0] + PetscSqrtReal(phys->gravity * u[0]); /*right wave*/
}

static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx, PetscInt m, const PetscScalar *u, PetscScalar *X, PetscScalar *Xi, PetscReal *speeds)
{
  PetscInt i, j;

  PetscFunctionBeginUser;
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) Xi[i * m + j] = X[i * m + j] = (PetscScalar)(i == j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx, PetscInt m, const PetscScalar *u, PetscScalar *X, PetscScalar *Xi, PetscReal *speeds)
{
  ShallowCtx *phys = (ShallowCtx *)vctx;
  PetscReal   c;
  PetscReal   tol = 1e-6;

  PetscFunctionBeginUser;
  c = PetscSqrtScalar(u[0] * phys->gravity);

  if (u[0] < tol) { /*Use conservative variables*/
    X[0 * 2 + 0] = 1;
    X[0 * 2 + 1] = 0;
    X[1 * 2 + 0] = 0;
    X[1 * 2 + 1] = 1;
    speeds[0]    = -c;
    speeds[1]    = c;
  } else {
    speeds[0]    = u[1] / u[0] - c;
    speeds[1]    = u[1] / u[0] + c;
    X[0 * 2 + 0] = 1;
    X[0 * 2 + 1] = speeds[0];
    X[1 * 2 + 0] = 1;
    X[1 * 2 + 1] = speeds[1];
  }

  PetscCall(PetscArraycpy(Xi, X, 4));
  PetscCall(PetscKernel_A_gets_inverse_A_2(Xi, 0, PETSC_FALSE, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsCharacteristic_Shallow_Mat(void *vctx, const PetscScalar *u, Mat eigmat)
{
  ShallowCtx *phys = (ShallowCtx *)vctx;
  PetscReal   c;
  PetscInt    m = 2, n = 2, i;
  PetscReal   X[m][n];
  PetscInt    idxm[m], idxn[n];

  PetscFunctionBeginUser;
  c = PetscSqrtScalar(u[0] * phys->gravity);

  for (i = 0; i < m; i++) idxm[i] = i;
  for (i = 0; i < n; i++) idxn[i] = i;
  /* Analytical formulation for the eigen basis of the Df for at u */
  X[0][0] = 1;
  X[1][0] = u[1] / u[0] - c;
  X[0][1] = 1;
  X[1][1] = u[1] / u[0] + c;
  PetscCall(MatSetValues(eigmat, m, idxm, n, idxn, (PetscReal *)X, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(eigmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(eigmat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsFluxDer_Shallow(void *vctx, const PetscReal *u, Mat jacobian)
{
  ShallowCtx *phys = (ShallowCtx *)vctx;
  PetscInt    m = 2, n = 2, i;
  PetscReal   X[m][n];
  PetscInt    idxm[m], idxn[n];

  PetscFunctionBeginUser;
  for (i = 0; i < m; i++) idxm[i] = i;
  for (i = 0; i < n; i++) idxn[i] = i;
  /* Analytical formulation for Df at u */
  X[0][0] = 0.;
  X[1][0] = -PetscSqr(u[1]) / PetscSqr(u[0]) + phys->gravity * u[0];
  X[0][1] = 1.;
  X[1][1] = 2. * u[1] / u[0];
  PetscCall(MatSetValues(jacobian, m, idxm, n, idxn, (PetscReal *)X, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsRoeAvg_Shallow(void *ctx, const PetscReal *uL, const PetscReal *uR, PetscReal *uavg)
{
  PetscFunctionBeginUser;
  uavg[0] = (uL[0] + uR[0]) / 2.0;
  uavg[1] = uavg[0] * (uL[1] / PetscSqrtReal(uL[0]) + uR[1] / PetscSqrtReal(uR[0])) / (PetscSqrtReal(uL[0]) + PetscSqrtReal(uR[0]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* For the SWE the Roe matrix can be computed by the Flux jacobian evaluated at a roe average point */
static PetscErrorCode PhysicsRoeMat_Shallow(void *ctx, const PetscReal *uL, const PetscReal *uR, Mat roe)
{
  PetscReal roeavg[2];

  PetscFunctionBeginUser;
  PetscCall(PhysicsRoeAvg_Shallow(ctx, uL, uR, roeavg));
  PetscCall(PhysicsFluxDer_Shallow(ctx, roeavg, roe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsSample_ShallowNetwork(void *vctx, PetscInt initial, PetscReal t, PetscReal x, PetscReal *u, PetscInt edgeid)
{
  ShallowCtx *phys = (ShallowCtx *)vctx;

  PetscFunctionBeginUser;
  if (t > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Exact solutions not implemented for t > 0");
  switch (initial) {
  case 0:
    if (edgeid == 0) {
      u[0] = phys->parenth;
      u[1] = phys->parentv;
    } else {
      u[0] = 1.0;
      u[1] = 0.0;
    }
    break;
  case 1: /* Initial 1-3 are from Jingmei's and Bennedito's paper */
    if (edgeid == 0) {
      u[0] = 0.5;
      u[1] = 0.1;
    } else if (edgeid == 1) {
      u[0] = 0.5;
      u[1] = 0.0;
    } else {
      u[0] = 1;
      u[1] = 0.0;
    }
    break;
  case 2:
    if (edgeid == 0) {
      u[0] = 1.0 + PetscExpReal(-20.0 * (x + 1.0) * (x + 1.0));
      u[1] = u[0] / 2.0;
    } else if (edgeid == 1) {
      u[0] = 1.0;
      u[1] = 0.0;
    } else {
      u[0] = 0.5;
      u[1] = 0.0;
    }
    break;
  case 3:
    if (edgeid == 0) {
      u[0] = ((x >= 0 && x <= 0.2) || (x >= 0.4 && x <= 0.6) || (x >= 0.8 && x <= 1.0)) ? 1.5 : 1.0;
      u[1] = u[0] / 5.0;
    } else if (edgeid == 1) {
      u[0] = 1.0;
      u[1] = 0.0;
    } else {
      u[0] = 0.5;
      u[1] = 0.0;
    }
    break;
  case 4:              /* Sunny's Test Case*/
    if (edgeid == 0) { /* Not sure what the correct IC is here*/
      u[0] = ((x >= 7 && x <= 9)) ? 2.0 - PetscSqr(x - 8) : 1.0;
      u[1] = 0.0;
    } else {
      u[0] = 1.0;
      u[1] = 0.0;
    }
    break;
  case 5: /* Roundabout Pulse */
    u[0] = !(edgeid % 2) ? 2 : 1;
    u[1] = 0;
    break;
    /* The following problems are based on geoemtrically 1d Networks, no notion of edgeid is considered */
  case 6:
    u[0] = (x < 10) ? 1 : 0.1;
    u[1] = (x < 10) ? 2.5 : 0;
    break;
  case 7:
    u[0] = (x < 25) ? 1 : 1;
    u[1] = (x < 25) ? -5 : 5;
    break;
  case 8:
    u[0] = (x < 20) ? 1 : 0;
    u[1] = (x < 20) ? 0 : 0;
    break;
  case 9:
    u[0] = (x < 30) ? 0 : 1;
    u[1] = (x < 30) ? 0 : 0;
    break;
  case 10:
    u[0] = (x < 25) ? 0.1 : 0.1;
    u[1] = (x < 25) ? -0.3 : 0.3;
    break;
  case 11:
    u[0] = 1 + 0.5 * PetscSinReal(2 * PETSC_PI * x);
    u[1] = 1 * u[0];
    break;
  case 12:
    u[0] = 1.0;
    u[1] = 1.0;
    break;
  case 13:
    u[0] = (x < -2) ? 2 : 1; /* Standard Dam Break Problem */
    u[1] = (x < -2) ? 0 : 0;
    break;
  case 14:
    u[0] = (x < 25) ? 2 : 1; /* Standard Dam Break Problem */
    u[1] = (x < 25) ? 0 : 0;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "unknown initial condition");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Lax Curve evaluation function, for use in RiemannSolver */
static PetscErrorCode LaxCurve_Shallow(RiemannSolver rs, const PetscReal *u, PetscReal hbar, PetscInt wavenumber, PetscReal *ubar)
{
  PetscReal   g, h, v;
  ShallowCtx *ctx;

  PetscFunctionBegin;
  PetscCall(RiemannSolverGetApplicationContext(rs, &ctx));
  g = ctx->gravity;
  h = u[0];
  v = u[1] / h;
  /* switch between the 1-wave and 2-wave curves */
  switch (wavenumber) {
  case 1:
    ubar[1] = hbar < h ? v - 2.0 * (PetscSqrtScalar(g * hbar) - PetscSqrtScalar(g * h)) : v - (hbar - h) * PetscSqrtScalar(g * (hbar + h) / (2.0 * hbar * h));
    ubar[1] *= hbar;
    break;
  case 2:
    ubar[1] = hbar < h ? v + 2.0 * (PetscSqrtScalar(g * hbar) - PetscSqrtScalar(g * h)) : v + (hbar - h) * PetscSqrtScalar(g * (hbar + h) / (2.0 * hbar * h));
    ubar[1] *= hbar;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Shallow Water Lax Curves have only 2 waves (1,2), requested wave number: %i \n", wavenumber);
    break;
  }
  ubar[0] = hbar;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysicsCreate_Shallow(DGNetwork fvnet)
{
  ShallowCtx *user;
  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user));
  fvnet->physics.samplenetwork  = PhysicsSample_ShallowNetwork;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.characteristic = PhysicsCharacteristic_Shallow;
  fvnet->physics.flux           = ShallowFlux;
  fvnet->physics.user           = user;
  fvnet->physics.dof            = 2;
  fvnet->physics.flux2          = ShallowFluxVoid;
  fvnet->physics.fluxeig        = ShallowEig;
  fvnet->physics.roeavg         = PhysicsRoeAvg_Shallow;
  fvnet->physics.eigbasis       = PhysicsCharacteristic_Shallow_Mat;
  fvnet->physics.fluxder        = PhysicsFluxDer_Shallow;
  fvnet->physics.roemat         = PhysicsRoeMat_Shallow;
  fvnet->physics.laxcurve       = LaxCurve_Shallow;
  PetscCall(PetscStrallocpy("height", &fvnet->physics.fieldname[0]));
  PetscCall(PetscStrallocpy("momentum", &fvnet->physics.fieldname[1]));
  user->gravity = 9.81;
  user->parenth = 2.0;
  user->parentv = 0.0;

  PetscOptionsBegin(fvnet->comm, fvnet->prefix, "Options for Shallow", "");
  PetscCall(PetscOptionsReal("-parh", "", "", user->parenth, &user->parenth, NULL));
  PetscCall(PetscOptionsReal("-parv", "", "", user->parentv, &user->parentv, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------- Traffic ----------------------------------- */

typedef struct {
  PetscReal a;
} TrafficCtx;

static inline PetscScalar TrafficChar(PetscScalar a, PetscScalar u)
{
  return a * (1 - 2 * u);
}

static inline PetscErrorCode TrafficFlux2(void *ctx, const PetscReal *u, PetscReal *f)
{
  PetscFunctionBeginUser;
  TrafficCtx *phys = (TrafficCtx *)ctx;
  f[0]             = phys->a * u[0] * (1. - u[0]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline void TrafficFluxVoid(void *ctx, const PetscReal *u, PetscReal *f)
{
  TrafficCtx *phys = (TrafficCtx *)ctx;
  f[0]             = phys->a * u[0] * (1. - u[0]);
}

static void TrafficEig(void *ctx, const PetscReal *u, PetscScalar *eig)
{
  PetscReal a = ((TrafficCtx *)ctx)->a;

  eig[0] = TrafficChar(a, u[0]);
}

typedef struct {
  PetscReal a, x, t;
} MethodCharCtx;

/* TODO Generalize to arbitrary initial value */
static PetscErrorCode TrafficCase1Char(SNES snes, Vec X, Vec f, void *ctx)
{
  PetscReal          x, t, rhs, a;
  const PetscScalar *s;

  PetscFunctionBeginUser;
  x = ((MethodCharCtx *)ctx)->x;
  t = ((MethodCharCtx *)ctx)->t;
  a = ((MethodCharCtx *)ctx)->a;

  PetscCall(VecGetArrayRead(X, &s));
  rhs = TrafficChar(a, PetscSinReal(PETSC_PI * (s[0] / 5.0)) + 2) * t + s[0] - x;
  PetscCall(VecSetValue(f, 0, rhs, INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO Generalize to arbitrary initial value */
static PetscErrorCode TrafficCase1Char_J(SNES snes, Vec X, Mat Amat, Mat Pmat, void *ctx)
{
  PetscReal          t, rhs, a;
  const PetscScalar *s;

  PetscFunctionBeginUser;
  t = ((MethodCharCtx *)ctx)->t;
  a = ((MethodCharCtx *)ctx)->a;

  PetscCall(VecGetArrayRead(X, &s));
  rhs = 1.0 - t * a * 2.0 * PETSC_PI / 5.0 * PetscCosReal(PETSC_PI * (s[0] / 5.0));
  PetscCall(MatSetValue(Pmat, 0, 0, rhs, INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &s));

  PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
  if (Amat != Pmat) {
    PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsSample_TrafficNetwork(void *vctx, PetscInt initial, PetscReal t, PetscReal x, PetscReal *u, PetscInt edgeid)
{
  SNES          snes;
  Mat           J;
  Vec           X, R;
  PetscReal    *s;
  MethodCharCtx ctx;

  PetscFunctionBeginUser;
  if (t < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "t must be >= 0 ");
  switch (initial) {
  case 0:
    if (t > 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Exact solution for case 0 not implemented for t > 0");
    if (edgeid == 0) {
      u[0] = (x < -2) ? 1.0 : 0.25; /* Traffic Break problem ?*/
    } else {
      u[0] = 0.5;
    }
    break;
  case 1:
    if (t == 0.0) {
      if (edgeid == 0) {
        u[0] = PetscSinReal(PETSC_PI * (x / 5.0));
      } else if (edgeid == 1) {
        u[0] = PetscSinReal(PETSC_PI * (-x / 5.0));
      } else {
        u[0] = 0;
      }
    } else {
      /* Method of characteristics to solve for exact solution */
      ctx.t = t;
      ctx.a = 0.5;
      /* this is a hack that assumes we are using network 6 the periodic network with discretizations of 
          x = [0,5] for both of them */
      ctx.x = !edgeid ? x : x + 5;
      PetscCall(VecCreate(PETSC_COMM_SELF, &X));
      PetscCall(VecSetSizes(X, PETSC_DECIDE, 1));
      PetscCall(VecSetFromOptions(X));
      PetscCall(VecDuplicate(X, &R));
      PetscCall(MatCreate(PETSC_COMM_SELF, &J));
      PetscCall(MatSetSizes(J, 1, 1, 1, 1));
      PetscCall(MatSetFromOptions(J));
      PetscCall(MatSetUp(J));
      PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
      PetscCall(SNESSetFunction(snes, R, TrafficCase1Char, &ctx));
      PetscCall(SNESSetJacobian(snes, J, J, TrafficCase1Char_J, &ctx));
      PetscCall(SNESSetFromOptions(snes));
      PetscCall(VecSet(X, x));
      PetscCall(SNESSolve(snes, NULL, X));
      PetscCall(VecGetArray(X, &s));
      u[0] = PetscSinReal(PETSC_PI * (s[0] / 5.0)) + 2;
      PetscCall(VecRestoreArray(X, &s));
      PetscCall(VecDestroy(&X));
      PetscCall(VecDestroy(&R));
      PetscCall(MatDestroy(&J));
      PetscCall(SNESDestroy(&snes));
    }
    break;
  case 2:
    if (edgeid == 0) {
      u[0] = 0.8;
    } else {
      u[0] = 0.0;
    }
    break;
  case 3:
    if (edgeid == 4) {
      if ((0 <= x && x <= 0.2) || (0.4 <= x && x <= 0.6) || (8.5 <= x && x <= 1.0)) {
        u[0] = 0.25;
      } else {
        u[0] = 0.35;
      }
    } else if (edgeid == 6) {
      u[0] = 0.2 + 0.2 * sin(5 * PETSC_PI * x);
    } else {
      u[0] = 0.5;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "unknown initial condition");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysicsFluxDer_Traffic(void *vctx, const PetscReal *u, Mat jacobian)
{
  TrafficCtx *traffic = (TrafficCtx *)vctx;

  PetscFunctionBeginUser;
  PetscCall(MatSetValue(jacobian, 0, 0, TrafficChar(traffic->a, u[0]), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysicsCreate_Traffic(DGNetwork fvnet)
{
  TrafficCtx *user;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user));
  fvnet->physics.samplenetwork  = PhysicsSample_TrafficNetwork;
  fvnet->physics.characteristic = PhysicsCharacteristic_Conservative;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.user           = user;
  fvnet->physics.dof            = 1;
  fvnet->physics.flux           = TrafficFlux2;
  fvnet->physics.flux2          = TrafficFluxVoid;
  fvnet->physics.fluxeig        = TrafficEig;
  fvnet->physics.fluxder        = PhysicsFluxDer_Traffic;

  PetscCall(PetscStrallocpy("density", &fvnet->physics.fieldname[0]));
  user->a = 4.0;
  PetscOptionsBegin(fvnet->comm, fvnet->prefix, "Options for Traffic", "");
  PetscCall(PetscOptionsReal("-physics_traffic_a", "Flux = a*u*(1-u)", "", user->a, &user->a, NULL));
  PetscOptionsEnd();

  PetscFunctionReturn(PETSC_SUCCESS);
}
