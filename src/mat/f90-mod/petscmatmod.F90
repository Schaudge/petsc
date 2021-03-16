
module petscmatdefdummy
  use petscvecdef
#include <../src/mat/f90-mod/petscmat.h>
end module petscmatdefdummy

module petscmatdef
  use petscmatdefdummy
  interface operator(.ne.)
     function matnotequal(A,B)
       import tMat
       logical matnotequal
       type(tMat), intent(in) :: A,B
     end function matnotequal
     function matfdcoloringnotequal(A,B)
       import tMatFDColoring
       logical matfdcoloringnotequal
       type(tMatFDColoring), intent(in) :: A,B
     end function matfdcoloringnotequal
     function matnullspacenotequal(A,B)
       import tMatNullSpace
       logical matnullspacenotequal
       type(tMatNullSpace), intent(in) :: A,B
     end function matnullspacenotequal
  end interface operator(.ne.)
  interface operator(.eq.)
     function matequals(A,B)
       import tMat
       logical matequals
       type(tMat), intent(in) :: A,B
     end function matequals
     function matfdcoloringequals(A,B)
       import tMatFDColoring
       logical matfdcoloringequals
       type(tMatFDColoring), intent(in) :: A,B
     end function matfdcoloringequals
     function matnullspaceequals(A,B)
       import tMatNullSpace
       logical matnullspaceequals
       type(tMatNullSpace), intent(in) :: A,B
     end function matnullspaceequals
  end interface operator(.eq.)
end module petscmatdef

function matnotequal(A,B)
  use petscmatdefdummy, only: tMat
  logical matnotequal
  type(tMat), intent(in) :: A,B
  matnotequal = (A%v .ne. B%v)
end function matnotequal

function matequals(A,B)
  use petscmatdefdummy, only: tMat
  logical matequals
  type(tMat), intent(in) :: A,B
  matequals = (A%v .eq. B%v)
end function matequals

function matfdcoloringnotequal(A,B)
  use petscmatdefdummy, only: tMatFDColoring
  logical matfdcoloringnotequal
  type(tMatFDColoring), intent(in) :: A,B
  matfdcoloringnotequal = (A%v .ne. B%v)
end function matfdcoloringnotequal

function matfdcoloringequals(A,B)
  use petscmatdefdummy, only: tMatFDColoring
  logical matfdcoloringequals
  type(tMatFDColoring), intent(in) :: A,B
  matfdcoloringequals = (A%v .eq. B%v)
end function matfdcoloringequals

function matnullspacenotequal(A,B)
  use petscmatdefdummy, only: tMatNullSpace
  logical matnullspacenotequal
  type(tMatNullSpace), intent(in) :: A,B
  matnullspacenotequal = (A%v .ne. B%v)
end function matnullspacenotequal

function matnullspaceequals(A,B)
  use petscmatdefdummy, only: tMatNullSpace
  logical matnullspaceequals
  type(tMatNullSpace), intent(in) :: A,B
  matnullspaceequals = (A%v .eq. B%v)
end function matnullspaceequals

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::matnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matequals
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringequals
!DEC$ ATTRIBUTES DLLEXPORT::matnullspacenotequal
!DEC$ ATTRIBUTES DLLEXPORT::matnullspaceequals
#endif
module petscmat
  use petscmatdef
  use petscvec
#include <../src/mat/f90-mod/petscmat.h90>
  interface
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
  end interface
end module petscmat

