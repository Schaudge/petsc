        module petscmldef
        use petsctaodef
#include <../src/ml/f90-mod/petscml.h>
        end module petscmldef

        module petscml
        use petscmldef
        use petsctao
        end module petscml

        module petscml
        use petscmldef
        use petsctao
#include <../src/ml/f90-mod/petscml.h90>
        interface
#include <../src/ml/f90-mod/ftn-auto-interfaces/petscml.h90>
        end interface
        end module

        module petscdef
        use petscdmdadef
        use petscdmplexdef
        use petscdmnetworkdef
        use petscdmpatchdef
        use petscdmforestdef
        use petscdmlabeldef
        use petsctsdef
        use petsctaodef
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
        end module petsc
