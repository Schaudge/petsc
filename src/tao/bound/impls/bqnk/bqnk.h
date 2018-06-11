/*
Context for bounded quasi-Newton-Krylov type optimization algorithms
*/

#if !defined(__TAO_BQNK_H)
#define __TAO_BQNK_H

#include <../src/tao/bound/impls/bnk/bnk.h>

typedef struct {
  Mat B;
  PC pc;
} TAO_BQNK;

#define BQNK_INIT_CONSTANT         0
#define BQNK_INIT_DIRECTION        1
#define BQNK_INIT_TYPES            2

static const char *BQNK_INIT[64] = {"constant", "direction", "interpolation"};

PETSC_INTERN PetscErrorCode TaoSetUp_BQNK(Tao);
PETSC_INTERN PetscErrorCode TaoCreate_BQNK(Tao);

#endif /* if !defined(__TAO_BQNK_H) */