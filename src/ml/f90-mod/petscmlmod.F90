        module petscmldef
        use petsctaodef
#include <../src/ml/f90-mod/petscml.h>
        end module petscmldef

        module petscml
        use petscmldef
        use petsctao
#include <../src/ml/f90-mod/petscml.h90>
        interface
#include <../src/ml/f90-mod/ftn-auto-interfaces/petscml.h90>
        end interface
        end module petscml

