#include "petsc/finclude/petscsnes.h"

      type tSNES
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tSNES

      type tPetscConvEst
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tPetscConvEst

      SNES, parameter :: PETSC_NULL_SNES = tSNES(0)
      PetscConvEst, parameter :: PETSC_NULL_CONVEST = tPetscConvEst(0)

      ! SNESConvergedReason
      PetscEnum, parameter :: SNES_CONVERGED_FNORM_ABS = 2
      PetscEnum, parameter :: SNES_CONVERGED_FNORM_RELATIVE = 3
      PetscEnum, parameter :: SNES_CONVERGED_SNORM_RELATIVE = 4
      PetscEnum, parameter :: SNES_CONVERGED_ITS = 5
      PetscEnum, parameter :: SNES_DIVERGED_FUNCTION_DOMAIN = -1
      PetscEnum, parameter :: SNES_DIVERGED_FUNCTION_COUNT = -2
      PetscEnum, parameter :: SNES_DIVERGED_LINEAR_SOLVE = -3
      PetscEnum, parameter :: SNES_DIVERGED_FNORM_NAN = -4
      PetscEnum, parameter :: SNES_DIVERGED_MAX_IT = -5
      PetscEnum, parameter :: SNES_DIVERGED_LINE_SEARCH = -6
      PetscEnum, parameter :: SNES_DIVERGED_INNER = -7
      PetscEnum, parameter :: SNES_DIVERGED_LOCAL_MIN = -8
      PetscEnum, parameter :: SNES_DIVERGED_DTOL = -9
      PetscEnum, parameter :: SNES_DIVERGED_JACOBIAN_DOMAIN = -10
      PetscEnum, parameter :: SNES_DIVERGED_TR_DELTA = -11
      PetscEnum, parameter :: SNES_CONVERGED_ITERATING = 0

      ! SNESNormSchedule
      PetscEnum, parameter :: SNES_NORM_DEFAULT = -1
      PetscEnum, parameter :: SNES_NORM_NONE = 0
      PetscEnum, parameter :: SNES_NORM_ALWAYS = 1
      PetscEnum, parameter :: SNES_NORM_INITIAL_ONLY = 2
      PetscEnum, parameter :: SNES_NORM_FINAL_ONLY = 3
      PetscEnum, parameter :: SNES_NORM_INITIAL_FINAL_ONLY = 4
      
      ! SNESLineSearchReason
      PetscEnum, parameter :: SNES_LINESEARCH_SUCCEEDED = 0
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_NANORINF = 1
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_DOMAIN = 2
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_REDUCT = 3
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_USER = 4
      PetscEnum, parameter :: SNES_LINESEARCH_FAILED_FUNCTION = 5

      ! SNESNGMRESRestartType
      PetscEnum, parameter :: SNES_NGMRES_RESTART_NONE = 0
      PetscEnum, parameter :: SNES_NGMRES_RESTART_PERIODIC = 1
      PetscEnum, parameter :: SNES_NGMRES_RESTART_DIFFERENCE = 2

      ! SNESNGMRESSelectType
      PetscEnum, parameter :: SNES_NGMRES_SELECT_NONE = 0
      PetscEnum, parameter :: SNES_NGMRES_SELECT_DIFFERENCE = 1
      PetscEnum, parameter :: SNES_NGMRES_SELECT_LINESEARCH = 2

      ! SNESNCGType
      PetscEnum, parameter :: SNES_NCG_FR = 0
      PetscEnum, parameter :: SNES_NCG_PRP = 1
      PetscEnum, parameter :: SNES_NCG_HS = 2
      PetscEnum, parameter :: SNES_NCG_DY = 3
      PetscEnum, parameter :: SNES_NCG_CD = 4

      ! SNESQNScaleType
      PetscEnum, parameter :: SNES_QN_SCALE_DEFAULT = 0
      PetscEnum, parameter :: SNES_QN_SCALE_NONE = 1
      PetscEnum, parameter :: SNES_QN_SCALE_SHANNO = 2
      PetscEnum, parameter :: SNES_QN_SCALE_LINESEARCH = 3
      PetscEnum, parameter :: SNES_QN_SCALE_JACOBIAN = 4

      ! SNESQNRestartType
      PetscEnum, parameter :: SNES_QN_RESTART_DEFAULT = 0
      PetscEnum, parameter :: SNES_QN_RESTART_NONE = 1
      PetscEnum, parameter :: SNES_QN_RESTART_POWELL = 2
      PetscEnum, parameter :: SNES_QN_RESTART_PERIODIC = 3

      ! SNESQNType
      PetscEnum, parameter :: SNES_QN_LBFGS = 0
      PetscEnum, parameter :: SNES_QN_BROYDEN = 1
      PetscEnum, parameter :: SNES_QN_BADBROYDEN = 2
