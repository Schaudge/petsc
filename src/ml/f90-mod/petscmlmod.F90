        module petscmldef
        use petsctaodef
#include <../src/ml/f90-mod/petscregressor.h>
        end module petscmldef

        module petscml
        use petscmldef
        use petsctao
#include <../src/ml/f90-mod/petscregressor.h90>
        interface
#include <../src/ml/f90-mod/ftn-auto-interfaces/petscregressor.h90>
        end interface
        end module petscml

! The all encompassing petsc module

        module petscdef
        use petscdmdadef
        use petscdmplexdef
        use petscdmnetworkdef
        use petscdmpatchdef
        use petscdmforestdef
        use petscdmlabeldef
        use petsctsdef
        use petsctaodef
        use petscmldef
        end module petscdef

        module petsc
        use petscdmda
        use petscdmplex
        use petscdmnetwork
        use petscdmpatch
        use petscdmforest
        use petscdmlabel
        use petscdt
        use petscts
        use petsctao
        use petscml
        end module petsc
