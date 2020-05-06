#if !defined (PETSCVECDEF_H)
#define PETSCVECDEF_H

#include "petsc/finclude/petscao.h"

#define Vec type(tVec)
#define VecScatter type(tVecScatter)
#define VecTagger type(tVecTagger)

#define VecType character*(80)
#define VecScatterType character*(80)

#define NormType PetscEnum
#define InsertMode PetscEnum
#define ScatterMode PetscEnum
#define VecOption PetscEnum

#define VecOperation PetscEnum
#define VecTaggerCDFMethod PetscEnum

#define VECSEQ         'seq'
#define VECMPI         'mpi'
#define VECSTANDARD    'standard'
#define VECSHARED      'shared'
#define VECSEQVIENNACL 'seqviennacl'
#define VECMPIVIENNACL 'mpiviennacl'
#define VECVIENNACL    'viennacl'
#define VECNEST        'nest'

#endif
