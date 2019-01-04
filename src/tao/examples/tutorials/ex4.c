static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>

typedef struct _UserCtx
{
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  NormType p;
}
* UserCtx;

int main (int argc, char** argv)
{
  UserCtx ctx;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

TEST*/
