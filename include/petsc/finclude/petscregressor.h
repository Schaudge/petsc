#if !defined (PETSCSNESDEF_H)
#define PETSCSNESDEF_H

#include "petsc/finclude/petsctao.h"

#define PetscRegressor type(tPetscRegressor)
#define PetscRegressorType character*(80)
#define PetscRegressorLinearType character*(80)

#define PETSCREGRESSORLINEAR 'linear'

#define PETSCREGRESSORLINEAROLS   'ols'
#define PETSCREGRESSORLINEARLASSO 'lasso'
#define PETSCREGRESSORLINEARRIDGE 'ridge'

#endif
