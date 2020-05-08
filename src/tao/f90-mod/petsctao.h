#include "petsc/finclude/petsctao.h"
      
      type tTao
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTao
      type tTaoLineSearch
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tTaoLineSearch

      Tao, parameter :: PETSC_NULL_TAO = tTao(0)
      TaoLineSearch, parameter :: PETSC_NULL_TAOLINESEARCH = tTaoLineSearch(0)

      ! TaoSubsetType
      PetscEnum, parameter :: TAO_SUBSET_SUBVEC = 0
      PetscEnum, parameter :: TAO_SUBSET_MASK = 1
      PetscEnum, parameter :: TAO_SUBSET_MATRIXFREE = 2

      ! TaoADMMUpdateType
      PetscEnum, parameter :: TAO_ADMM_UPDATE_BASIC = 0
      PetscEnum, parameter :: TAO_ADMM_UPDATE_ADAPTIVE = 1
      PetscEnum, parameter :: TAO_ADMM_UPDATE_ADAPTIVE_RELAXED = 2

      ! TaoADMMRegularizerType
      PetscEnum, parameter :: TAO_ADMM_REGULARIZER_USER = 0
      PetscEnum, parameter :: TAO_ADMM_REGULARIZER_SOFT_THRESH = 1

      ! TaoConvergedReason
      PetscEnum, parameter :: TAO_CONVERGED_GATOL = 3
      PetscEnum, parameter :: TAO_CONVERGED_GRTOL = 4
      PetscEnum, parameter :: TAO_CONVERGED_GTTOL = 5
      PetscEnum, parameter :: TAO_CONVERGED_STEPTOL = 6
      PetscEnum, parameter :: TAO_CONVERGED_MINF = 7
      PetscEnum, parameter :: TAO_CONVERGED_USER = 8
      PetscEnum, parameter :: TAO_DIVERGED_MAXITS = -2
      PetscEnum, parameter :: TAO_DIVERGED_NAN = -4
      PetscEnum, parameter :: TAO_DIVERGED_MAXFCN = -5
      PetscEnum, parameter :: TAO_DIVERGED_LS_FAILURE = -6
      PetscEnum, parameter :: TAO_DIVERGED_TR_REDUCTION = -7
      PetscEnum, parameter :: TAO_DIVERGED_USER = -8
      PetscEnum, parameter :: TAO_CONTINUE_ITERATING = 0

      ! TaoLineSearchConvergedReason
      PetscEnum, parameter :: TAOLINESEARCH_FAILED_INFORNAN = -1
      PetscEnum, parameter :: TAOLINESEARCH_FAILED_BADPARAMETER = -2
      PetscEnum, parameter :: TAOLINESEARCH_FAILED_ASCENT = -3
      PetscEnum, parameter :: TAOLINESEARCH_CONTINUE_ITERATING = 0
      PetscEnum, parameter :: TAOLINESEARCH_SUCCESS = 1
      PetscEnum, parameter :: TAOLINESEARCH_SUCCESS_USER = 2
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_OTHER = 3
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_MAXFCN = 4
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_UPPERBOUND = 5
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_LOWERBOUND = 6
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_RTOL = 7
      PetscEnum, parameter :: TAOLINESEARCH_HALTED_USER = 8
