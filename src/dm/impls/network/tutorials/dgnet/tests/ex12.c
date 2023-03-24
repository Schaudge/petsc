static const char help[] = "Silly test for NetRP, to be made a proper test later inside that directory \n\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscriemannsolver.h>
#include <petscnetrp.h>
#include <petsc/private/kernels/blockinvert.h>

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

int main(int argc, char *argv[])
{
  MPI_Comm      comm;
  PetscMPIInt   size, rank;
  Vec           U, UStar;
  RiemannSolver flux;
  NetRP         netrp, netrplinear;
  PetscScalar  *u;
  PetscBool     edgein[3];

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
  PetscCall(NetRPSetType(netrp, NETRPEXACTSWE));
  PetscCall(NetRPSetFlux(netrp, flux));

  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrplinear));
  PetscCall(NetRPSetType(netrplinear, NETRPLINEARIZED));
  PetscCall(NetRPSetFlux(netrplinear, flux));
  PetscCall(RiemannSolverDestroy(&flux));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 6, &U));
  PetscCall(VecDuplicate(U, &UStar));

  /* set riemann data */
  PetscCall(VecGetArray(U, &u));
  u[0] = 2.0;
  u[1] = 0.0;
  u[2] = 1.5;
  u[3] = 0.0;
  u[4] = 1.0;
  u[5] = 1.0;
  PetscCall(VecRestoreArray(U, &u));

  /* set riemann problem topology */
  edgein[0] = PETSC_TRUE;
  edgein[1] = PETSC_FALSE;
  edgein[2] = PETSC_FALSE;

  PetscCall(NetRPSolveStar(netrp, 1,2, edgein, U, UStar));

  PetscCall(PetscPrintf(comm, "Riemann Data  \n \n"));
  PetscCall(VecView(U, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(comm, "Exact SWE Solve \n \n"));
  PetscCall(VecView(UStar, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(NetRPSolveStar(netrplinear, 1,2, edgein, U, UStar));
  PetscCall(PetscPrintf(comm, "Linearized SWE Solve \n \n"));
  PetscCall(VecView(UStar, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(NetRPDestroy(&netrp));
  PetscCall(NetRPDestroy(&netrplinear));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&UStar));

  PetscCall(PetscFinalize());
}
