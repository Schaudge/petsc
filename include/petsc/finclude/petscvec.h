!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (PETSCVECDEF_H)
#define PETSCVECDEF_H

#include "petsc/finclude/petscao.h"

#define Vec type(tVec)
#define VecScatter type(tVecScatter)
#define VecTagger type(tVecTagger)

#define NormType PetscEnum
#define InsertMode PetscEnum
#define ScatterMode PetscEnum
#define VecOption PetscEnum
#define VecType character*(80)
#define VecOperation PetscEnum
#define VecTaggerCDFMethod PetscEnum

#define VECSEQ 'seq'
#define VECMPI 'mpi'
#define VECSTANDARD 'standard'
#define VECSHARED 'shared'
#define VECSEQCUDA 'seq:cuda'
#define VECMPICUDA 'mpi:cuda'
#define VECCUDA    'standard:cuda'
#define VECSEQVIENNACL 'seq:viennacl'
#define VECMPIVIENNACL 'mpi:viennacl'
#define VECVIENNACL 'standard:viennacl'
#define VECNEST 'nest'

#define VecScatterType character*(80)

#endif
