
        module petscdmswarmdef
        use petscdmdef
#include <../src/dm/f90-mod/petscdmswarm.h>
        end module

        module petscdmswarm
        use petscdmswarmdef
#include <../src/dm/f90-mod/petscdmswarm.h90>
        interface
#include <../ftn-auto/dm/f90-mod/petscdmswarm.h90>
        end interface
        end module
