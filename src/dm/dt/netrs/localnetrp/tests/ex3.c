#include "mpi.h"
#include "petscerror.h"
#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
static const char help[] = "Test NetRPTrafic\n\n";

#include <petscriemannsolver.h>
#include <petscnetrp.h>

typedef struct {
  PetscReal a;
} TrafficCtx;

static inline PetscScalar TrafficChar(PetscScalar a, PetscScalar u)
{
  return a * (1 - 2 * u);
}

static inline void TrafficFluxVoid(void *ctx, const PetscReal *u, PetscReal *f)
{
  PetscReal a = 4.0;
  f[0]        = a * u[0] * (1. - u[0]);
}

static void TrafficEig(void *ctx, const PetscReal *u, PetscScalar *eig)
{
  PetscReal a = 4.0;
  eig[0]      = TrafficChar(a, u[0]);
}

static PetscErrorCode PhysicsFluxDer_Traffic(void *vctx, const PetscReal *u, Mat jacobian)
{
  PetscReal a = 4.0;

  PetscFunctionBeginUser;
  PetscCall(MatSetValue(jacobian, 0, 0, TrafficChar(a, u[0]), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(jacobian, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jacobian, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TrafficDistribution(NetRP rp, PetscInt indeg, PetscInt outdeg, Mat distribution)
{
  PetscScalar *mat;

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArray(distribution, &mat));
  PetscCheck(indeg == 2 && outdeg == 2, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER, "Only have traffic distribution matrix for indeg 2 and outdeg 2 for now");
  /* 2x2 matrix from benedettos book */
  mat[0 * indeg + 0] = 1. / 3.;
  mat[1 * indeg + 0] = 1. / 4.;
  mat[0 * indeg + 1] = 2. / 3.;
  mat[1 * indeg + 1] = 3. / 4.; /*? dense are stored column oriented ? */
  PetscCall(MatDenseRestoreArray(distribution, &mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode TrafficDistribution2(NetRP rp, PetscInt indeg, PetscInt outdeg, Mat distribution)
{
  PetscScalar *mat;
  PetscInt     i, j;
  PetscReal    val = 1. / (outdeg);

  PetscFunctionBeginUser;
  PetscCall(MatDenseGetArray(distribution, &mat));
  PetscCheck(indeg == outdeg, PetscObjectComm((PetscObject)rp), PETSC_ERR_USER, "Only have traffic distribution matrix for indeg == outdeg  for now");
  /* equal distribution */
  for (i = 0; i < outdeg; i++) {
    for (j = 0; j < indeg; j++) { mat[i * indeg + j] = val; }
  }
  PetscCall(MatDenseRestoreArray(distribution, &mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode NetRPViewRiemannProblem(NetRP rp, Vec U, Vec Flux)
{
  MPI_Comm    comm;
  PetscMPIInt rank;
  PetscReal   sigma;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)rp, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF), "--Riemann Problem--\n"));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF), "Riemann Problem Parameters:\n"));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(NetRPTrafficGetFluxMaximumPoint(rp, &sigma));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF), "sigma: %e \n", sigma));
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF), "Riemann Data\n"));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(VecView(U, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF), "Flux\n"));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(VecView(Flux, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));

    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  MPI_Comm      comm;
  PetscMPIInt   size, rank;
  RiemannSolver flux;
  NetRP         netrp;
  PetscInt      indeg = 2, outdeg = 2, vdeg, i, num_in_u, num_out_u;
  PetscScalar  *u;
  PetscBool    *edgein;
  Vec           U, UCopy, Flux;

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscOptionsBegin(comm, NULL, "ex3 options", "NetRP");
  PetscCall(PetscOptionsInt("-indeg", "Number of edges directed into the Riemann Problem", "ex3.c", indeg, &indeg, NULL));
  PetscCall(PetscOptionsInt("-outdeg", "Number of edges directed out of the Riemann Problem", "ex3.c", outdeg, &outdeg, NULL));
  PetscOptionsEnd();

  /* Create Vector for storing Riemann Data */
  vdeg = indeg + outdeg;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, vdeg, &U));
  PetscCall(VecDuplicate(U, &Flux));

  /* Topology of the problem */
  PetscCall(PetscMalloc1(vdeg, &edgein));
  for (i = 0; i < indeg; i++) edgein[i] = PETSC_TRUE;
  for (i = indeg; i < vdeg; i++) edgein[i] = PETSC_FALSE;

  /* Command line setting of Riemann Data */
  PetscCall(VecGetArray(U, &u));
  for (i = 0; i < vdeg; i++) u[i] = 1.0;

  num_in_u  = indeg;
  num_out_u = outdeg;

  PetscOptionsBegin(comm, NULL, "ex3 options", "NetRP");
  PetscCall(PetscOptionsScalarArray("-edgein_data", "Riemann Data for in edges", "ex3.c", u, &num_in_u, NULL));
  PetscCall(PetscOptionsScalarArray("-edgeout_data", "Riemann Data for out edges", "ex3.c", u + indeg, &num_out_u, NULL));
  PetscOptionsEnd();

  PetscCall(VecRestoreArray(U, &u));
  PetscCall(VecDuplicate(U, &UCopy));
  PetscCall(VecCopy(U, UCopy));

  /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to
   set all the physics parts at once) */
  PetscCall(RiemannSolverCreate(comm, &flux));
  PetscCall(RiemannSolverSetFromOptions(flux));
  PetscCall(RiemannSolverSetFluxEig(flux, TrafficEig));
  PetscCall(RiemannSolverSetFlux(flux, 1, 1, TrafficFluxVoid));
  PetscCall(RiemannSolverSetJacobian(flux, PhysicsFluxDer_Traffic));
  PetscCall(RiemannSolverSetUp(flux));

  /* Set up NetRP */
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrp));
  PetscCall(NetRPSetType(netrp, NETRPTRAFFICLWR));
  PetscCall(NetRPSetFlux(netrp, flux));
  PetscCall(RiemannSolverDestroy(&flux));
  PetscCall(NetRPTrafficSetDistribution(netrp, TrafficDistribution));
  PetscCall(NetRPSetUp(netrp));

  PetscCall(NetRPSolveFlux(netrp, indeg, outdeg, edgein, U, Flux));

  PetscCall(VecGetArray(U, &u));
  u[0] = 0.1464466094;
  u[1] = 0.75;
  u[2] = 0.25;
  u[3] = 0.853553;
  PetscCall(VecRestoreArray(U, &u));

  PetscCall(NetRPSolveFlux(netrp, indeg, outdeg, edgein, U, Flux));
  PetscCall(NetRPSolveFlux(netrp, indeg, outdeg, edgein, UCopy, Flux));

  PetscCall(PetscFree(edgein));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&UCopy));
  PetscCall(VecDestroy(&Flux));
  PetscCall(NetRPDestroy(&netrp));
  PetscCall(PetscFinalize());
}
/*TEST

  testset:
    nsize: 1
    test:
      suffix: bennedetto_0
      args: -indeg 2 -outdeg 2 -edgein_data ,0.75 -edgeout_data 0.25,0.853553
TEST*/