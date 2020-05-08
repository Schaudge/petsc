#if !defined (PETSCDMPLEXDEF_H)
#define PETSCDMPLEXDEF_H

#include "petsc/finclude/petscdm.h"
#include "petsc/finclude/petscdmlabel.h"

#define DMPlexInterpolatedFlag PetscEnum
#define DMPlexCellRefinerType PetscEnum

#define PETSCPARTITIONERCHACO    'chaco'
#define PETSCPARTITIONERPARMETIS 'parmetis'
#define PETSCPARTITIONERPTSCOTCH 'ptscotch'
#define PETSCPARTITIONERSHELL    'shell'
#define PETSCPARTITIONERSIMPLE   'simple'
#define PETSCPARTITIONERGATHER   'gather'
#define PETSCPARTITIONERMATPARTITIONING 'matpartitioning'

#endif
