!
!  MAT_INFO_SIZE must equal # elements in MatInfo structure
!  (See petsc/include/petscmat.h)
!  This is needed in f90 interface for MatGetInfo() - hence
!  in a separate include

      PetscEnum, parameter :: MAT_INFO_SIZE=10
