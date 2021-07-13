#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

/*
  "Full" memory implementation of only the diagonal terms in a symmetric Broyden approximation.
*/

typedef struct {
  Vec invDnew, invD, temp, temp2;    /* work vectors for diagonal scaling */
  PetscReal tol, lip, mu;            /* Lipschitz constant threshold and mu weight for changing tendency of Hessian */
  PetscBool allocated;
} Mat_DiagBB;
