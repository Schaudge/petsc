#if !defined(MLREGRESSORIMPL_H)
#define MLREGRESSORIMPL_H

#include <petscmlregressor.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool MLRegressorRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MLRegressorRegisterAll(void);

typedef struct _MLRegressorOps *MRegressorOps;

struct _MLRegressorOps {
  PetscErrorCode (*setup)(MLRegressor);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,MLRegressor);         /* sets options from database */
  PetscErrorCode (*settraining)(MLRegressor,Mat,Vec);                      /* set the training data matrix and targets */
  PetscErrorCode (*fit)(MLRegressor);                                      /* compute the transformation to be applied */
  PetscErrorCode (*predict)(MLRegressor,Mat,Vec);                          /* predict using fitted model */
  PetscErrorCode (*destroy)(MLRegressor);
  PetscErrorCode (*reset)(MLRegressor);
  PetscErrorCode (*view)(MLRegressor,PetscViewer);
};

/* Define the MLRegressor data structure. */
struct _p_MLRegressor {
  PETSCHEADER(struct _MLRegressorOps);

  PetscBool setupcalled;  /* True if setup has been called */
  void      *data;        /* Implementation-specific data */
  Mat       training;     /* Matrix holding the training data set */
  Vec       target;       /* Targets for training data (response variables or labels) */
};

#endif
