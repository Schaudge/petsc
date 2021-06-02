#if !defined(MLDRIMPL_H)
#define MLDRIMPL_H

#include <petscmldr.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool MLDRRegisterAllCalled;
PETSC_EXTERN PetscErrorCode MLDRRegisterAll(void);

typedef struct _MLDROps *MLDROps;

struct _MLDROps {
  PetscErrorCode (*setup)(MLDR);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,MLDR);         /* sets options from database */
  PetscErrorCode (*fit)(MLDR);                                      /* compute the transformation to be applied */
  PetscErrorCode (*transform)(MLDR,Mat,Mat*);                       /* apply computed transformation to matrix */
  PetscErrorCode (*destroy)(MLDR);
  PetscErrorCode (*reset)(MLDR);
};

/* Define the MLDR data structure. */
struct _p_MLDR {
  PETSCHEADER(struct _MLDROps);

  PetscBool setupcalled; /* True if setup has been called */
  void      *data; /* Implementation-specific data */
  Mat training;  /* Matrix holding the training data set */
};

#endif
