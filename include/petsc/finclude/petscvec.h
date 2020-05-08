#if !defined (PETSCVECDEF_H)
#define PETSCVECDEF_H

#include "petsc/finclude/petscao.h"

#define Vec type(tVec)
#define VecScatter type(tVecScatter)
#define VecTagger type(tVecTagger)

#define VecType character*(80)
#define VecScatterType character*(80)
#define VecTaggerType character*(80)

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
#define VECSEQCUDA     'seqcuda'
#define VECMPICUDA     'mpicuda'
#define VECCUDA        'cuda'
#define VECNEST        'nest'
#define VECNODE        'node'
#define VECSCATTERSEQ       'seq'
#define VECSCATTERMPI1      'mpi1'
#define VECSCATTERMPI3      'mpi3'
#define VECSCATTERMPI3NODE  'mpi3node'
#define VECSCATTERSF        'sf'
#define VECTAGGERABSOLUTE   'absolute'
#define VECTAGGERRELATIVE   'relative'
#define VECTAGGERCDF        'cdf'
#define VECTAGGEROR         'or'
#define VECTAGGERAND        'and'

#endif
