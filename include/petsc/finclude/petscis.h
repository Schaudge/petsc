#if !defined (PETSCISDEF_H)
#define PETSCISDEF_H

#include "petsc/finclude/petscsys.h"
#include "petsc/finclude/petscviewer.h"

#define IS type(tIS)
#define ISColoring type(tISColoring)
#define PetscSection type(tPetscSection)
#define PetscSectionSym type(tPetscSectionSym)
#define PetscSF type(tPetscSF)

#define ISType character*(80)
#define ISGlobalToLocalType character*(80)

#define PetscLayout PetscFortranAddr
#define ISInfo PetscEnum
#define ISInfoType PetscEnum
#define ISLocalToGlobalMapping PetscFortranAddr
#define ISGlobalToLocalMappingMode PetscEnum
#define ISColoringType PetscEnum

#define ISColoringValue PETSC_IS_COLOR_VALUE_TYPE_F

#define ISGENERAL 'general'
#define ISSTRIDE 'stride'
#define ISBLOCK 'block'

#define ISGLOBALTOLOCALMAPPINGBASIC 'basic'
#define ISGLOBALTOLOCALMAPPINGHASH  'hash'
#endif
