/* 
    Currently just a dumping ground for physics functions needed for the various tests. Namely flux functions 
    eigenvalues, characteristic decompositions, initial condition specifications, exact solutions, 
    network riemann solvers (to be removed as class is built for them specifically)
*/

#include "fluxfun.h"

// static inline PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }

/* Default Destruction Function */

static inline PetscErrorCode FluxFunDestroy_Default(FluxFunction *flux)
{
  PetscInt field;

  PetscFunctionBeginUser;
  PetscCall(PetscFree((*flux)->user));
  for (field = 0; field < (*flux)->dof; field++) { PetscCall(PetscFree((*flux)->fieldname[field])); }
  *flux = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------- Shallow Water ----------------------------------- */
typedef struct {
  PetscReal gravity;
} ShallowCtx;

// static inline PetscErrorCode ShallowFlux(void *ctx, const PetscReal *u, PetscReal *f)
// {
//   ShallowCtx *phys = (ShallowCtx *)ctx;
//   f[0]             = u[1];
//   f[1]             = PetscSqr(u[1]) / u[0] + 0.5 * phys->gravity * PetscSqr(u[0]);
//   PetscFunctionReturn(PETSC_SUCCESS);
// }
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

PetscErrorCode PhysicsCreate_Shallow(FluxFunction *fluxfun)
{
  ShallowCtx  *user;
  FluxFunction flux;

  PetscFunctionBeginUser;
  *fluxfun = NULL;

  PetscCall(PetscNew(&user));
  PetscCall(PetscNew(&flux));
  user->gravity = 9.81;
  flux->dof     = 2;
  PetscCall(PetscStrallocpy("height", &flux->fieldname[0]));
  PetscCall(PetscStrallocpy("momentum", &flux->fieldname[1]));

  flux->eigbasis = PhysicsCharacteristic_Shallow_Mat;
  flux->roeavg   = PhysicsRoeAvg_Shallow;
  flux->flux     = ShallowFluxVoid;
  flux->fluxder  = PhysicsFluxDer_Shallow;
  flux->fluxeig  = ShallowEig;
  flux->user     = user;
  flux->roemat   = PhysicsRoeMat_Shallow;
  flux->laxcurve = LaxCurve_Shallow;
  flux->destroy  = FluxFunDestroy_Default;

  *fluxfun = flux;
  PetscFunctionReturn(PETSC_SUCCESS);
}
