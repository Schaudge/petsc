static const char help[] = "Test SolverCtx for NetRP\n\n";

#include <petscriemannsolver.h>
#include <petscnetrp.h>

static inline void ShallowFluxVoid(void *ctx, const PetscReal *u, PetscReal *f)
{
  PetscReal gravity = 9.81;
  f[0]              = u[1];
  f[1]              = PetscSqr(u[1]) / u[0] + 0.5 * gravity * PetscSqr(u[0]);
}

static inline void ShallowEig(void *ctx, const PetscReal *u, PetscReal *eig)
{
  PetscReal gravity = 9.81;
  eig[0]            = u[1] / u[0] - PetscSqrtReal(gravity * u[0]); /*left wave*/
  eig[1]            = u[1] / u[0] + PetscSqrtReal(gravity * u[0]); /*right wave*/
}

static PetscErrorCode PhysicsCharacteristic_Shallow_Mat(void *vctx, const PetscScalar *u, Mat eigmat)
{
  PetscReal c, gravity = 9.81;
  PetscInt  m = 2, n = 2, i;
  PetscReal X[m][n];
  PetscInt  idxm[m], idxn[n];

  PetscFunctionBeginUser;
  c = PetscSqrtScalar(u[0] * gravity);

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

/* Lax Curve evaluation function, for use in RiemannSolver */
static PetscErrorCode LaxCurve_Shallow(RiemannSolver rs, const PetscReal *u, PetscReal hbar, PetscInt wavenumber, PetscReal *ubar)
{
  PetscReal g = 9.81, h, v;

  PetscFunctionBegin;
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

typedef struct {
  PetscInt  id;
  PetscReal x;
} userCtx;

static PetscErrorCode UserNetRPSetSolverCtx(NetRP rp, PetscInt vdegin, PetscInt vdegout, void **ctx)
{
  userCtx *solverctx;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(1, &solverctx));
  solverctx->id = vdegin;
  solverctx->x  = PETSC_PI * vdegout;

  *ctx = (void *)solverctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UserNetRPDestroySolverCtx(NetRP rp, PetscInt vdegin, PetscInt vdegout, void *ctx)
{
  userCtx *solverctx = (userCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(solverctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode PrintDebugInfo(MPI_Comm comm, PetscInt indeg, PetscInt outdeg, userCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm, "(indeg,outdeg) : ( %" PetscInt_FMT ", %" PetscInt_FMT " ) \n    Solver Ctx \n      id : %" PetscInt_FMT "\n      x  : %f \n", indeg, outdeg, ctx->id, ctx->x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  MPI_Comm      comm;
  PetscMPIInt   size, rank;
  RiemannSolver flux;
  NetRP         netrp, netrpdir;
  userCtx      *userctx;
  PetscInt      indeg[3], outdeg[3];

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to
   set all the physics parts at once) */
  PetscCall(RiemannSolverCreate(comm, &flux));
  PetscCall(RiemannSolverSetFromOptions(flux));
  PetscCall(RiemannSolverSetFluxEig(flux, ShallowEig));
  PetscCall(RiemannSolverSetEigBasis(flux, PhysicsCharacteristic_Shallow_Mat));
  PetscCall(RiemannSolverSetFlux(flux, 1, 2, ShallowFluxVoid));
  PetscCall(RiemannSolverSetLaxCurve(flux, LaxCurve_Shallow));
  PetscCall(RiemannSolverSetUp(flux));

  /* Set up NetRP */
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrp));
  PetscCall(NetRPSetType(netrp, NETRPBLANK));
  PetscCall(NetRPSetFlux(netrp, flux));
  PetscCall(NetRPSetPhysicsGenerality(netrp, Generic));
  PetscCall(NetRPSetCacheType(netrp, UndirectedVDeg));

  PetscCall(NetRPDuplicate(netrp, &netrpdir));
  PetscCall(NetRPSetCacheType(netrpdir, DirectedVDeg));

  PetscCall(RiemannSolverDestroy(&flux));

  /* Set Solver Ctx */
  PetscCall(NetRPSetSolverCtxFunc(netrp, UserNetRPSetSolverCtx));
  PetscCall(NetRPSetDestroySolverCtxFunc(netrp, UserNetRPDestroySolverCtx));

  PetscCall(NetRPSetSolverCtxFunc(netrpdir, UserNetRPSetSolverCtx));
  PetscCall(NetRPSetDestroySolverCtxFunc(netrpdir, UserNetRPDestroySolverCtx));

  PetscCall(NetRPSetUp(netrp));
  PetscCall(NetRPSetUp(netrpdir));

  /* Cache Solvers for NetRP */
  indeg[0]  = 0;
  outdeg[0] = 2;
  indeg[1]  = 2;
  outdeg[1] = 4;
  indeg[2]  = 3;
  outdeg[2] = 1;

  /* check if ctx creation works */
  PetscCall(NetRPCacheSolvers(netrp, 3, indeg, outdeg));
  PetscCall(NetRPCacheSolvers(netrpdir, 3, indeg, outdeg));

  PetscCall(NetRPGetSolverCtx(netrp, 0, 2, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 0, 2, userctx));
  PetscCall(NetRPGetSolverCtx(netrp, 6, 0, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 2, 4, userctx));
  PetscCall(NetRPGetSolverCtx(netrp, 0, 4, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 3, 1, userctx));

  PetscCall(NetRPGetSolverCtx(netrpdir, 0, 2, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 0, 2, userctx));
  PetscCall(NetRPGetSolverCtx(netrpdir, 2, 4, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 2, 4, userctx));
  PetscCall(NetRPGetSolverCtx(netrpdir, 3, 1, (void **)&userctx));
  PetscCall(PrintDebugInfo(PETSC_COMM_WORLD, 3, 1, userctx));

  PetscCall(NetRPDestroy(&netrp));
  PetscCall(NetRPDestroy(&netrpdir));
  PetscCall(PetscFinalize());
}
