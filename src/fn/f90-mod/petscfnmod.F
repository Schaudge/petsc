#define PETSC_USE_FORTRAN_MODULES

        module petscfndefdummy
        use petscmatdef
#include <../src/fn/f90-mod/petscfn.h>
        end module

        module petscfndef
        use petscfndefdummy
        interface operator(.ne.)
          function fnnotequal(A,B)
            use petscfndefdummy
            logical fnnotequal
            type(tPetscFn), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function fnequals(A,B)
            use petscfndefdummy
            logical fnequals
            type(tPetscFn), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function fnnotequal(A,B)
          use petscfndefdummy
          logical fnnotequal
          type(tPetscFn), intent(in) :: A,B
          fnnotequal = (A%v .ne. B%v)
        end function

       function fnequals(A,B)
          use petscfndefdummy
          logical fnequals
          type(tPetscFn), intent(in) :: A,B
          fnequals = (A%v .eq. B%v)
        end function

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::fnnotequal
!DEC$ ATTRIBUTES DLLEXPORT::fnequals
#endif
        module petscfn
        use petscfndef
        use petscmat
        use petscvec
#include <../src/fn/f90-mod/petscfn.h90>
        interface
#include <../src/fn/f90-mod/ftn-auto-interfaces/petscfn.h90>
        end interface
        end module

