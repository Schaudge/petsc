#include "petsc/finclude/petsctao.h"

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
