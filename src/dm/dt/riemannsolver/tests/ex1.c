static char help[] = "Tests repeated RiemannSolverSetType().\n\n";

#include <petscriemannsolver.h>

/* Dummy functions to force the RiemannSolver to generate all internal structures */
static PetscErrorCode EigBasis_Dummy(void *ctx, const PetscReal *u, Mat mat)
{
  PetscFunctionReturn(0);
}

static PetscErrorCode RoeMat_Dummy(void *ctx, const PetscReal *uL, const PetscReal *uR, Mat *roe)
{
  PetscFunctionReturn(0);
}

static PetscErrorCode RoeAvg_Dummy(void *ctx, const PetscReal *uL, const PetscReal *uR, PetscReal *roeavg)
{
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  PetscInt       dim = 1, numfields = 5;
  RiemannSolver  rs;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* create RiemannSolver */
  PetscCall(RiemannSolverCreate(PETSC_COMM_SELF, &rs));
  PetscCall(RiemannSolverSetFluxDim(rs, dim, numfields));
  PetscCall(RiemannSolverSetEigBasis(rs, EigBasis_Dummy));
  PetscCall(RiemannSolverSetRoeMatrixFunct(rs, RoeMat_Dummy));
  PetscCall(RiemannSolverSetRoeAvgFunct(rs, RoeAvg_Dummy));
  PetscCall(RiemannSolverSetType(rs, "lax"));
  PetscCall(RiemannSolverSetUp(rs));

  /* Add a different type here when I have another implementation coded up */
  PetscCall(RiemannSolverSetType(rs, "lax"));
  PetscCall(RiemannSolverSetUp(rs));

  ierr = RiemannSolverDestroy(&rs);
  PetscCall(PetscFinalize());
  return 0;
}
