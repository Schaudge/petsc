#include "petsc/finclude/petscdm.h"

      type tDM
        sequence
        PetscFortranAddr :: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDM
      type tDMPlexCellRefiner
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tDMPlexCellRefiner

      DM, parameter :: PETSC_NULL_DM = tDM(0)
      DMPlexCellRefiner, parameter :: PETSC_NULL_DMPLEXCELLREFINER = tDMPlexCellRefiner(0)


      ! DMBoundaryType
      PetscEnum, parameter :: DM_BOUNDARY_NONE = 0
      PetscEnum, parameter :: DM_BOUNDARY_GHOSTED = 1
      PetscEnum, parameter :: DM_BOUNDARY_MIRROR = 2
      PetscEnum, parameter :: DM_BOUNDARY_PERIODIC = 3
      PetscEnum, parameter :: DM_BOUNDARY_TWIST = 4

      ! DMBoundaryConditionType
      PetscEnum, parameter :: DM_BC_ESSENTIAL = 1
      PetscEnum, parameter :: DM_BC_ESSENTIAL_FIELD = 5
      PetscEnum, parameter :: DM_BC_NATURAL = 2
      PetscEnum, parameter :: DM_BC_NATURAL_FIELD = 6
      PetscEnum, parameter :: DM_BC_ESSENTIAL_BD_FIELD = 9
      PetscEnum, parameter :: DM_BC_NATURAL_RIEMANN = 10

      ! DMPointLocationType
      PetscEnum, parameter :: DM_POINTLOCATION_NONE = 0
      PetscEnum, parameter :: DM_POINTLOCATION_NEAREST = 1
      PetscEnum, parameter :: DM_POINTLOCATION_REMOVE = 2

      ! DMAdaptationStrategy
      PetscEnum, parameter :: DM_ADAPTATION_INITIAL = 0
      PetscEnum, parameter :: DM_ADAPTATION_SEQUENTIAL = 1
      PetscEnum, parameter :: DM_ADAPTATION_MULTILEVEL = 2

      ! DMAdaptationCriterion
      PetscEnum, parameter :: DM_ADAPTATION_NONE = 0
      PetscEnum, parameter :: DM_ADAPTATION_REFINE = 1
      PetscEnum, parameter :: DM_ADAPTATION_LABEL = 2
      PetscEnum, parameter :: DM_ADAPTATION_METRIC = 3

      ! DMAdaptFlag
      PetscEnum, parameter :: DM_ADAPT_DETERMINE=-1
      PetscEnum, parameter :: DM_ADAPT_KEEP = 0
      PetscEnum, parameter :: DM_ADAPT_REFINE = 1
      PetscEnum, parameter :: DM_ADAPT_COARSEN = 2
      PetscEnum, parameter :: DM_ADAPT_RESERVED_COUNT=3

      ! DMDirection
      PetscEnum, parameter :: DM_X = 0
      PetscEnum, parameter :: DM_Y = 1
      PetscEnum, parameter :: DM_Z = 2

      ! DMEnclosureType
      PetscEnum, parameter :: DM_ENC_EQUALITY = 0
      PetscEnum, parameter :: DM_ENC_SUPERMESH = 1
      PetscEnum, parameter :: DM_ENC_SUBMESH = 2
      PetscEnum, parameter :: DM_ENC_NONE = 3
      PetscEnum, parameter :: DM_ENC_UNKNOWN = 4

      ! DMPolytopeType
      PetscEnum, parameter :: DM_POLYTOPE_POINT = 0
      PetscEnum, parameter :: DM_POLYTOPE_SEGMENT = 1
      PetscEnum, parameter :: DM_POLYTOPE_POINT_PRISM_TENSOR = 2
      PetscEnum, parameter :: DM_POLYTOPE_TRIANGLE = 3
      PetscEnum, parameter :: DM_POLYTOPE_QUADRILATERAL = 4
      PetscEnum, parameter :: DM_POLYTOPE_SEG_PRISM_TENSOR = 5
      PetscEnum, parameter :: DM_POLYTOPE_TETRAHEDRON = 6
      PetscEnum, parameter :: DM_POLYTOPE_HEXAHEDRON = 7
      PetscEnum, parameter :: DM_POLYTOPE_TRI_PRISM = 8
      PetscEnum, parameter :: DM_POLYTOPE_TRI_PRISM_TENSOR = 9
      PetscEnum, parameter :: DM_POLYTOPE_QUAD_PRISM_TENSOR = 10
      PetscEnum, parameter :: DM_POLYTOPE_FV_GHOST = 11
      PetscEnum, parameter :: DM_POLYTOPE_INTERIOR_GHOST = 12
      PetscEnum, parameter :: DM_POLYTOPE_UNKNOWN = 13
      PetscEnum, parameter :: DM_NUM_POLYTOPES = 14

      ! PetscUnit
      PetscEnum, parameter :: PETSC_UNIT_LENGTH = 0
      PetscEnum, parameter :: PETSC_UNIT_MASS = 1
      PetscEnum, parameter :: PETSC_UNIT_TIME = 2
      PetscEnum, parameter :: PETSC_UNIT_CURRENT = 3
      PetscEnum, parameter :: PETSC_UNIT_TEMPERATURE = 4
      PetscEnum, parameter :: PETSC_UNIT_AMOUNT = 5
      PetscEnum, parameter :: PETSC_UNIT_LUMINOSITY = 6
      PetscEnum, parameter :: NUM_PETSC_UNITS = 7

