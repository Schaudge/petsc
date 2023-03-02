static const char help[] = "Error Test for the Linearized method for the Shallow Water test \n\n";

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
  MatAssemblyBegin(eigmat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(eigmat, MAT_FINAL_ASSEMBLY);
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PetscErrorCode VecViewCSV_6(Vec X, PetscViewer viewer) 
{
  PetscScalar *x; 
  PetscFunctionBegin; 
  PetscCall(VecGetArray(X,&x));
  PetscCall(PetscViewerASCIIPrintf(viewer,"%e, %e, %e, %e, %e, %e \n",x[0],x[1],x[2],x[3],x[4],x[5])); 
  PetscCall(VecRestoreArray(X,&x));
  PetscFunctionReturn(0); 
}

int main(int argc, char *argv[])
{
  MPI_Comm      comm;
  PetscMPIInt   size, rank;
  Vec           U,UStarlin, UStar,Error; 
  RiemannSolver flux;
  NetRP         netrplin, netrpexact; 
  PetscScalar  *u;
  PetscBool    *edgein;
  PetscInt      i,ii,jj, j, kk,k, numsolves = 1e2, numedges = 3; /* This is required for now */
  PetscReal     parenthmin = 1.0, parenthmax = 2.0, parentqmin = -3.0, parentqmax = 3, g = 9.81, deltah, deltaq;
  PetscViewer   UView, UStarView, ULinView, ErrView; 

  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscOptionsBegin(comm, NULL, "ex13 options", "");
  PetscCall(PetscOptionsReal("-parenth_min", "[hmin,hmax] is the range of h values the parent edge takes", "", parenthmin, &parenthmin, NULL));
  PetscCall(PetscOptionsReal("-parenth_max", "[hmin,hmax] is the range of h values the parent edge takes", "", parenthmax, &parenthmax, NULL));
  PetscCall(PetscOptionsReal("-parentq_min", "[qmin,qmax] is the range of q values the parent edge takes", "", parentqmin, &parentqmin, NULL));
  PetscCall(PetscOptionsReal("-parentq_max", "[qmin,qmax] is the range of q values the parent edge takes", "", parentqmax, &parentqmax, NULL));
  PetscCall(PetscOptionsInt("-numsolves", "Number of NetRP solves per range. So total is numsolves^nedges ", "", numsolves, &numsolves, NULL));
  //PetscCall(PetscOptionsInt("-nedges", "Number of Edges in the star graph", "", numedges, &numedges, NULL));
  PetscOptionsEnd();

  /* input checks */
  PetscCheck(parenthmax - parenthmin >= 0, comm, PETSC_ERR_USER_INPUT, "parenth_max must be larger than parenth_min");
  PetscCheck(parentqmax - parentqmin >= 0, comm, PETSC_ERR_USER_INPUT, "parentq_max must be larger than parentq_min");
  PetscCheck(parenthmax > 0, comm, PETSC_ERR_USER_INPUT, "parenth_max must be greater than 0");
  PetscCheck(parenthmin > 0, comm, PETSC_ERR_USER_INPUT, "parenth_min must be greater than 0");
  /* fluvial check, make sure no initial data is in the fluvial regime, where |q| >= sqrt(g*h)*/
  PetscCheck(PetscAbs(parentqmax) < PetscSqrtReal(g * parenthmin) && PetscAbs(parentqmin) < PetscSqrtReal(g * parenthmin), comm, PETSC_ERR_USER_INPUT, "Fluvial Conditions detected in [hmin,hmax] X [qmin,qmax]. Change this domain");

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
  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrpexact));
  PetscCall(NetRPSetType(netrpexact, NETRPEXACTSWE));
  PetscCall(NetRPSetFlux(netrpexact, flux));
  PetscCall(NetRPSetFromOptions(netrpexact));

  PetscCall(NetRPCreate(PETSC_COMM_SELF, &netrplin));
  PetscCall(NetRPSetType(netrplin, NETRPLINEARIZED));
  PetscCall(NetRPSetFlux(netrplin, flux));
  PetscCall(NetRPSetFromOptions(netrplin));

  PetscCall(RiemannSolverDestroy(&flux));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 2 * numedges, &U));
  PetscCall(VecDuplicate(U, &UStar));
  PetscCall(VecDuplicate(U, &UStarlin));
  PetscCall(VecDuplicate(U, &Error));

  /* Create Viewers */
  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &UView)); 
  PetscCall(PetscViewerSetType(UView,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetName(UView,"UData.csv"));
  PetscCall(PetscViewerFileSetMode(UView,FILE_MODE_APPEND)); 

  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &UStarView));
  PetscCall(PetscViewerSetType(UStarView,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetName(UStarView,"UStarData.csv"));
  PetscCall(PetscViewerFileSetMode(UStarView,FILE_MODE_APPEND)); 

  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &ULinView));
  PetscCall(PetscViewerSetType(ULinView,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetName(ULinView,"ULinData.csv"));
  PetscCall(PetscViewerFileSetMode(ULinView,FILE_MODE_APPEND)); 

  PetscCall(PetscViewerCreate(PETSC_COMM_SELF, &ErrView));
  PetscCall(PetscViewerSetType(ErrView,PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetName(ErrView,"ErrData.csv"));
  PetscCall(PetscViewerFileSetMode(ErrView,FILE_MODE_APPEND)); 

  /* set riemann problem topology */
  PetscCall(PetscMalloc1(numedges, &edgein));
  edgein[0] = PETSC_TRUE;
  for (i = 1; i < numedges; i++) edgein[i] = PETSC_FALSE;

  /* perform a solve with a uniform range in [hmin,hmax] X [qmin,qmax] */
  deltah = (parenthmax - parenthmin) / (numsolves-1);
  deltaq = (parentqmax - parentqmin) / (numsolves-1);
  
  /* 6 order tensor product in phase space (automate?)*/
  for (i=0; i < numsolves; i++) {
    for(ii=0; ii < numsolves; ii++) {
      for(j=0; j < numsolves; j++) {
        for(jj=0; jj < numsolves; jj++) { 
          for(k=0; k <numsolves; k++) {
            for (kk=0; kk < numsolves; kk++) {
              PetscCall(VecGetArray(U, &u));
                u[0] = parenthmin + i * deltah;
                u[1] = parentqmin + ii * deltaq;
                u[2] = parenthmin + j * deltah;
                u[3] = parentqmin + jj * deltaq;
                u[4] = parenthmin + k * deltah;
                u[5] = parentqmin + kk * deltaq;                
              PetscCall(VecRestoreArray(U, &u));
              PetscCall(NetRPSolveStar(netrplin, numedges, edgein, U, UStarlin));
              PetscCall(NetRPSolveStar(netrpexact, numedges, edgein, U, UStar));

              PetscCall(VecCopy(UStar,Error));
              PetscCall(VecAXPY(Error,-1,UStarlin));

              PetscCall(VecViewCSV_6(U,UView)); 
              PetscCall(VecViewCSV_6(UStarlin,ULinView));
              PetscCall(VecViewCSV_6(UStar,UStarView)); 
              PetscCall(VecViewCSV_6(Error,ErrView)); 
            }
          }
        }
      }
    }
  }

  PetscCall(PetscFree(edgein));
  PetscCall(NetRPDestroy(&netrplin));
  PetscCall(NetRPDestroy(&netrpexact));

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&UStar));
  PetscCall(VecDestroy(&UStarlin));
  PetscCall(VecDestroy(&Error));

  PetscCall(PetscViewerDestroy(&UView));
  PetscCall(PetscViewerDestroy(&ULinView));
  PetscCall(PetscViewerDestroy(&UStarView));
  PetscCall(PetscViewerDestroy(&ErrView));

  PetscCall(PetscFinalize());
}