static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>

typedef struct _UserCtx
{
  PetscInt m;      /* The row dimension of F */
  PetscInt n;      /* The column dimension of F */
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  NormType p;
} * UserCtx;

PetscErrorCode ConfigureContext(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ctx->m = 10;
  ctx->n = 10;
  ctx->alpha = 1.;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The regularization multiplier", "ex4.c", ctx->alpha, &(ctx->alpha), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char** argv)
{
  UserCtx ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = ConfigureContext(ctx);CHKERRQ(ierr);
  /* build the matrix F in ctx */
  /* build the rhs d in ctx */
  /* Define two functions that could pass as objectives to TaoSetObjectiveRoutine(): one
   * for the misfit component, and one for the regularization component */

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */

  /* Construct the Tao object */

  /* solve */

  /* examine solution */

  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

TEST*/
