!
!  Used by petscmatmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscmat.h"

      type tMat
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tMat
      type tMatNullSpace
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tMatNullSpace
      type tMatFDColoring
        sequence
        PetscFortranAddr:: v PETSC_FORTRAN_TYPE_INITIALIZE
      end type tMatFDColoring

      Mat, parameter :: PETSC_NULL_MAT = tMat(0)
      MatFDColoring, parameter :: PETSC_NULL_MATFDCOLORING = tMatFDColoring(0)
      MatNullSpace, parameter :: PETSC_NULL_MATNULLSPACE = tMatNullSpace(0)
!
!  Flag for matrix assembly
!
      PetscEnum, parameter :: MAT_FLUSH_ASSEMBLY=1
      PetscEnum, parameter :: MAT_FINAL_ASSEMBLY=0
!
!
!
      PetscEnum, parameter :: MAT_FACTOR_NONE=0
      PetscEnum, parameter :: MAT_FACTOR_LU=1
      PetscEnum, parameter :: MAT_FACTOR_CHOLESKY=2
      PetscEnum, parameter :: MAT_FACTOR_ILU=3
      PetscEnum, parameter :: MAT_FACTOR_ICC=4
      PetscEnum, parameter :: MAT_FACTOR_ILUDT=5
      PetscEnum, parameter :: MAT_FACTOR_QR=6
!
! MatCreateSubMatrixOption
!
      PetscEnum, parameter :: MAT_DO_NOT_GET_VALUES=0
      PetscEnum, parameter :: MAT_GET_VALUES=1
!
!  MatOption; must match those in include/petscmat.h
!
      PetscEnum, parameter :: MAT_OPTION_MIN = -3
      PetscEnum, parameter :: MAT_UNUSED_NONZERO_LOCATION_ERR = -2
      PetscEnum, parameter :: MAT_ROW_ORIENTED = -1
      PetscEnum, parameter :: MAT_SYMMETRIC = 1
      PetscEnum, parameter :: MAT_STRUCTURALLY_SYMMETRIC = 2
      PetscEnum, parameter :: MAT_FORCE_DIAGONAL_ENTRIES = 3
      PetscEnum, parameter :: MAT_IGNORE_OFF_PROC_ENTRIES = 4
      PetscEnum, parameter :: MAT_USE_HASH_TABLE = 5
      PetscEnum, parameter :: MAT_KEEP_NONZERO_PATTERN = 6
      PetscEnum, parameter :: MAT_IGNORE_ZERO_ENTRIES = 7
      PetscEnum, parameter :: MAT_USE_INODES = 8
      PetscEnum, parameter :: MAT_HERMITIAN = 9
      PetscEnum, parameter :: MAT_SYMMETRY_ETERNAL = 10
      PetscEnum, parameter :: MAT_NEW_NONZERO_LOCATION_ERR = 11
      PetscEnum, parameter :: MAT_IGNORE_LOWER_TRIANGULAR = 12
      PetscEnum, parameter :: MAT_ERROR_LOWER_TRIANGULAR = 13
      PetscEnum, parameter :: MAT_GETROW_UPPERTRIANGULAR = 14
      PetscEnum, parameter :: MAT_SPD = 15
      PetscEnum, parameter :: MAT_NO_OFF_PROC_ZERO_ROWS = 16
      PetscEnum, parameter :: MAT_NO_OFF_PROC_ENTRIES = 17
      PetscEnum, parameter :: MAT_NEW_NONZERO_LOCATIONS = 18
      PetscEnum, parameter :: MAT_NEW_NONZERO_ALLOCATION_ERR = 19
      PetscEnum, parameter :: MAT_SUBSET_OFF_PROC_ENTRIES = 20
      PetscEnum, parameter :: MAT_SUBMAT_SINGLEIS = 21
      PetscEnum, parameter :: MAT_STRUCTURE_ONLY = 22
      PetscEnum, parameter :: MAT_SORTED_FULL = 23
      PetscEnum, parameter :: MAT_FORM_EXPLICIT_TRANSPOSE = 24
      PetscEnum, parameter :: MAT_STRUCTURAL_SYMMETRY_ETERNAL = 25
      PetscEnum, parameter :: MAT_SPD_ETERNAL = 26
      PetscEnum, parameter :: MAT_OPTION_MAX = 27
!
!  MatFactorShiftType
!
      PetscEnum, parameter :: MAT_SHIFT_NONE=0
      PetscEnum, parameter :: MAT_SHIFT_NONZERO=1
      PetscEnum, parameter :: MAT_SHIFT_POSITIVE_DEFINITE=2
      PetscEnum, parameter :: MAT_SHIFT_INBLOCKS=3
!
!  MatFactorError
!
      PetscEnum, parameter :: MAT_FACTOR_NOERROR=0
      PetscEnum, parameter :: MAT_FACTOR_STRUCT_ZEROPIVOT=1
      PetscEnum, parameter :: MAT_FACTOR_NUMERIC_ZEROPIVOT=2
      PetscEnum, parameter :: MAT_FACTOR_OUTMEMORY=3
      PetscEnum, parameter :: MAT_FACTOR_OTHER=4
!
!  MatDuplicateOption
!
      PetscEnum, parameter :: MAT_DO_NOT_COPY_VALUES=0
      PetscEnum, parameter :: MAT_COPY_VALUES=1
      PetscEnum, parameter :: MAT_SHARE_NONZERO_PATTERN=2
!
!  Flags for MatCopy, MatAXPY
!
      PetscEnum, parameter :: DIFFERENT_NONZERO_PATTERN = 0
      PetscEnum, parameter :: SUBSET_NONZERO_PATTERN = 1
      PetscEnum, parameter :: SAME_NONZERO_PATTERN = 2
      PetscEnum, parameter :: UNKNOWN_NONZERO_PATTERN = 3

#include "../src/mat/f90-mod/petscmatinfosize.h"

      PetscEnum, parameter :: MAT_INFO_BLOCK_SIZE=1
      PetscEnum, parameter :: MAT_INFO_NZ_ALLOCATED=2
      PetscEnum, parameter :: MAT_INFO_NZ_USED=3
      PetscEnum, parameter :: MAT_INFO_NZ_UNNEEDED=4
      PetscEnum, parameter :: MAT_INFO_MEMORY=5
      PetscEnum, parameter :: MAT_INFO_ASSEMBLIES=6
      PetscEnum, parameter :: MAT_INFO_MALLOCS=7
      PetscEnum, parameter :: MAT_INFO_FILL_RATIO_GIVEN=8
      PetscEnum, parameter :: MAT_INFO_FILL_RATIO_NEEDED=9
      PetscEnum, parameter :: MAT_INFO_FACTOR_MALLOCS=10
!
!  MatReuse
!
      PetscEnum, parameter :: MAT_INITIAL_MATRIX=0
      PetscEnum, parameter :: MAT_REUSE_MATRIX=1
      PetscEnum, parameter :: MAT_IGNORE_MATRIX=2
      PetscEnum, parameter :: MAT_INPLACE_MATRIX=3
!
!  MatInfoType
!
      PetscEnum, parameter :: MAT_LOCAL=1
      PetscEnum, parameter :: MAT_GLOBAL_MAX=2
      PetscEnum, parameter :: MAT_GLOBAL_SUM=3

!
!  MatCompositeType
!
      PetscEnum, parameter :: MAT_COMPOSITE_ADDITIVE = 0
      PetscEnum, parameter :: MAT_COMPOSITE_MULTIPLICATIVE = 1

#include "../src/mat/f90-mod/petscmatfactorinfosize.h"

      PetscEnum, parameter :: MAT_FACTORINFO_DIAGONAL_FILL = 1
      PetscEnum, parameter :: MAT_FACTORINFO_USEDT = 2
      PetscEnum, parameter :: MAT_FACTORINFO_DT = 3
      PetscEnum, parameter :: MAT_FACTORINFO_DTCOL = 4
      PetscEnum, parameter :: MAT_FACTORINFO_DTCOUNT = 5
      PetscEnum, parameter :: MAT_FACTORINFO_FILL = 6
      PetscEnum, parameter :: MAT_FACTORINFO_LEVELS = 7
      PetscEnum, parameter :: MAT_FACTORINFO_PIVOT_IN_BLOCKS = 8
      PetscEnum, parameter :: MAT_FACTORINFO_ZERO_PIVOT = 9
      PetscEnum, parameter :: MAT_FACTORINFO_SHIFT_TYPE = 10
      PetscEnum, parameter :: MAT_FACTORINFO_SHIFT_AMOUNT = 11
!
!  Options for SOR and SSOR
!  MatSorType may be bitwise ORd together, so do not change the numbers
!
      PetscEnum, parameter :: SOR_FORWARD_SWEEP=1
      PetscEnum, parameter :: SOR_BACKWARD_SWEEP=2
      PetscEnum, parameter :: SOR_SYMMETRIC_SWEEP=3
      PetscEnum, parameter :: SOR_LOCAL_FORWARD_SWEEP=4
      PetscEnum, parameter :: SOR_LOCAL_BACKWARD_SWEEP=8
      PetscEnum, parameter :: SOR_LOCAL_SYMMETRIC_SWEEP=12
      PetscEnum, parameter :: SOR_ZERO_INITIAL_GUESS=16
      PetscEnum, parameter :: SOR_EISENSTAT=32
      PetscEnum, parameter :: SOR_APPLY_UPPER=64
      PetscEnum, parameter :: SOR_APPLY_LOWER=128
!
!  MatOperation
!
      PetscEnum, parameter :: MATOP_SET_VALUES=0
      PetscEnum, parameter :: MATOP_GET_ROW=1
      PetscEnum, parameter :: MATOP_RESTORE_ROW=2
      PetscEnum, parameter :: MATOP_MULT=3
      PetscEnum, parameter :: MATOP_MULT_ADD=4
      PetscEnum, parameter :: MATOP_MULT_TRANSPOSE=5
      PetscEnum, parameter :: MATOP_MULT_TRANSPOSE_ADD=6
      PetscEnum, parameter :: MATOP_SOLVE=7
      PetscEnum, parameter :: MATOP_SOLVE_ADD=8
      PetscEnum, parameter :: MATOP_SOLVE_TRANSPOSE=9
      PetscEnum, parameter :: MATOP_SOLVE_TRANSPOSE_ADD=10
      PetscEnum, parameter :: MATOP_LUFACTOR=11
      PetscEnum, parameter :: MATOP_CHOLESKYFACTOR=12
      PetscEnum, parameter :: MATOP_SOR=13
      PetscEnum, parameter :: MATOP_TRANSPOSE=14
      PetscEnum, parameter :: MATOP_GETINFO=15
      PetscEnum, parameter :: MATOP_EQUAL=16
      PetscEnum, parameter :: MATOP_GET_DIAGONAL=17
      PetscEnum, parameter :: MATOP_DIAGONAL_SCALE=18
      PetscEnum, parameter :: MATOP_NORM=19
      PetscEnum, parameter :: MATOP_ASSEMBLY_BEGIN=20
      PetscEnum, parameter :: MATOP_ASSEMBLY_END=21
      PetscEnum, parameter :: MATOP_SET_OPTION=22
      PetscEnum, parameter :: MATOP_ZERO_ENTRIES=23
      PetscEnum, parameter :: MATOP_ZERO_ROWS=24
      PetscEnum, parameter :: MATOP_LUFACTOR_SYMBOLIC=25
      PetscEnum, parameter :: MATOP_LUFACTOR_NUMERIC=26
      PetscEnum, parameter :: MATOP_CHOLESKY_FACTOR_SYMBOLIC=27
      PetscEnum, parameter :: MATOP_CHOLESKY_FACTOR_NUMERIC=28
      PetscEnum, parameter :: MATOP_SETUP=29
      PetscEnum, parameter :: MATOP_ILUFACTOR_SYMBOLIC=30
      PetscEnum, parameter :: MATOP_ICCFACTOR_SYMBOLIC=31
      PetscEnum, parameter :: MATOP_GET_DIAGONAL_BLOCK=32
      PetscEnum, parameter :: MATOP_SET_INF=33
      PetscEnum, parameter :: MATOP_DUPLICATE=34
      PetscEnum, parameter :: MATOP_FORWARD_SOLVE=35
      PetscEnum, parameter :: MATOP_BACKWARD_SOLVE=36
      PetscEnum, parameter :: MATOP_ILUFACTOR=37
      PetscEnum, parameter :: MATOP_ICCFACTOR=38
      PetscEnum, parameter :: MATOP_AXPY=39
      PetscEnum, parameter :: MATOP_CREATE_SUBMATRICES=40
      PetscEnum, parameter :: MATOP_INCREASE_OVERLAP=41
      PetscEnum, parameter :: MATOP_GET_VALUES=42
      PetscEnum, parameter :: MATOP_COPY=43
      PetscEnum, parameter :: MATOP_GET_ROW_MAX=44
      PetscEnum, parameter :: MATOP_SCALE=45
      PetscEnum, parameter :: MATOP_SHIFT=46
      PetscEnum, parameter :: MATOP_DIAGONAL_SET=47
      PetscEnum, parameter :: MATOP_ZERO_ROWS_COLUMNS=48
      PetscEnum, parameter :: MATOP_SET_RANDOM=49
      PetscEnum, parameter :: MATOP_GET_ROW_IJ=50
      PetscEnum, parameter :: MATOP_RESTORE_ROW_IJ=51
      PetscEnum, parameter :: MATOP_GET_COLUMN_IJ=52
      PetscEnum, parameter :: MATOP_RESTORE_COLUMN_IJ=53
      PetscEnum, parameter :: MATOP_FDCOLORING_CREATE=54
      PetscEnum, parameter :: MATOP_COLORING_PATCH=55
      PetscEnum, parameter :: MATOP_SET_UNFACTORED=56
      PetscEnum, parameter :: MATOP_PERMUTE=57
      PetscEnum, parameter :: MATOP_SET_VALUES_BLOCKED=58
      PetscEnum, parameter :: MATOP_CREATE_SUBMATRIX=59
      PetscEnum, parameter :: MATOP_DESTROY=60
      PetscEnum, parameter :: MATOP_VIEW=61
      PetscEnum, parameter :: MATOP_CONVERT_FROM=62
      PetscEnum, parameter :: MATOP_IS_REAL=63
      PetscEnum, parameter :: MATOP_MATMAT_MULT_SYMBOLIC=64
      PetscEnum, parameter :: MATOP_MATMAT_MULT_NUMERIC=65
      PetscEnum, parameter :: MATOP_SET_LOCAL_TO_GLOBAL_MAP=66
      PetscEnum, parameter :: MATOP_SET_VALUES_LOCAL=67
      PetscEnum, parameter :: MATOP_ZERO_ROWS_LOCAL=68
      PetscEnum, parameter :: MATOP_GET_ROW_MAX_ABS=69
      PetscEnum, parameter :: MATOP_GET_ROW_MIN_ABS=70
      PetscEnum, parameter :: MATOP_CONVERT=71
      PetscEnum, parameter :: MATOP_HAS_OPERATION=72
      PetscEnum, parameter :: MATOP_PLACEHOLDER_73=73
      PetscEnum, parameter :: MATOP_SET_VALUES_ADIFOR=74
      PetscEnum, parameter :: MATOP_FD_COLORING_APPLY=75
      PetscEnum, parameter :: MATOP_SET_FROM_OPTIONS=76
      PetscEnum, parameter :: MATOP_PLACEHOLDER_77=77
      PetscEnum, parameter :: MATOP_PLACEHOLDER_78=78
      PetscEnum, parameter :: MATOP_FIND_ZERO_DIAGONALS=79
      PetscEnum, parameter :: MATOP_MULT_MULTIPLE=80
      PetscEnum, parameter :: MATOP_SOLVE_MULTIPLE=81
      PetscEnum, parameter :: MATOP_GET_INERTIA=82
      PetscEnum, parameter :: MATOP_LOAD=83
      PetscEnum, parameter :: MATOP_IS_SYMMETRIC=84
      PetscEnum, parameter :: MATOP_IS_HERMITIAN=85
      PetscEnum, parameter :: MATOP_IS_STRUCTURALLY_SYMMETRIC=86
      PetscEnum, parameter :: MATOP_SET_VALUES_BLOCKEDLOCAL=87
      PetscEnum, parameter :: MATOP_CREATE_VECS=88
      PetscEnum, parameter :: MATOP_PLACEHOLDER_89=89
      PetscEnum, parameter :: MATOP_MAT_MULT_SYMBOLIC=90
      PetscEnum, parameter :: MATOP_MAT_MULT_NUMERIC=91
      PetscEnum, parameter :: MATOP_PLACEHOLDER_92=92
      PetscEnum, parameter :: MATOP_PTAP_SYMBOLIC=93
      PetscEnum, parameter :: MATOP_PTAP_NUMERIC=94
      PetscEnum, parameter :: MATOP_PLACEHOLDER_95=95
      PetscEnum, parameter :: MATOP_MAT_TRANSPOSE_MULT_SYMBO=96
      PetscEnum, parameter :: MATOP_MAT_TRANSPOSE_MULT_NUMER=97
      PetscEnum, parameter :: MATOP_BIND_TO_CPU=98
      PetscEnum, parameter :: MATOP_PRODUCTSETFROMOPTIONS=99
      PetscEnum, parameter :: MATOP_PRODUCTSYMBOLIC=100
      PetscEnum, parameter :: MATOP_PRODUCTNUMERIC=101
      PetscEnum, parameter :: MATOP_CONJUGATE=102
      PetscEnum, parameter :: MATOP_VIEW_NATIVE=103
      PetscEnum, parameter :: MATOP_SET_VALUES_ROW=104
      PetscEnum, parameter :: MATOP_REAL_PART=105
      PetscEnum, parameter :: MATOP_IMAGINARY_PART=106
      PetscEnum, parameter :: MATOP_GET_ROW_UPPER_TRIANGULAR=107
      PetscEnum, parameter :: MATOP_RESTORE_ROW_UPPER_TRIANG=108
      PetscEnum, parameter :: MATOP_MAT_SOLVE=109
      PetscEnum, parameter :: MATOP_MAT_SOLVE_TRANSPOSE=110
      PetscEnum, parameter :: MATOP_GET_ROW_MIN=111
      PetscEnum, parameter :: MATOP_GET_COLUMN_VECTOR=112
      PetscEnum, parameter :: MATOP_MISSING_DIAGONAL=113
      PetscEnum, parameter :: MATOP_GET_SEQ_NONZERO_STRUCTUR=114
      PetscEnum, parameter :: MATOP_CREATE=115
      PetscEnum, parameter :: MATOP_GET_GHOSTS=116
      PetscEnum, parameter :: MATOP_GET_LOCAL_SUB_MATRIX=117
      PetscEnum, parameter :: MATOP_RESTORE_LOCALSUB_MATRIX=118
      PetscEnum, parameter :: MATOP_MULT_DIAGONAL_BLOCK=119
      PetscEnum, parameter :: MATOP_HERMITIAN_TRANSPOSE=120
      PetscEnum, parameter :: MATOP_MULT_HERMITIAN_TRANSPOSE=121
      PetscEnum, parameter :: MATOP_MULT_HERMITIAN_TRANS_ADD=122
      PetscEnum, parameter :: MATOP_GET_MULTI_PROC_BLOCK=123
      PetscEnum, parameter :: MATOP_FIND_NONZERO_ROWS=124
      PetscEnum, parameter :: MATOP_GET_COLUMN_NORMS=125
      PetscEnum, parameter :: MATOP_INVERT_BLOCK_DIAGONAL=126
      PetscEnum, parameter :: MATOP_INVERT_VBLOCK_DIAGONAL=127
      PetscEnum, parameter :: MATOP_CREATE_SUB_MATRICES_MPI=128
      PetscEnum, parameter :: MATOP_SET_VALUES_BATCH=129
      PetscEnum, parameter :: MATOP_PLACEHOLDER_130=130
      PetscEnum, parameter :: MATOP_TRANSPOSE_MAT_MULT_SYMBO=131
      PetscEnum, parameter :: MATOP_TRANSPOSE_MAT_MULT_NUMER=132
      PetscEnum, parameter :: MATOP_TRANSPOSE_COLORING_CREAT=133
      PetscEnum, parameter :: MATOP_TRANS_COLORING_APPLY_SPT=134
      PetscEnum, parameter :: MATOP_TRANS_COLORING_APPLY_DEN=135
      PetscEnum, parameter :: MATOP_PLACEHOLDER_136=136
      PetscEnum, parameter :: MATOP_RART_SYMBOLIC=137
      PetscEnum, parameter :: MATOP_RART_NUMERIC=138
      PetscEnum, parameter :: MATOP_SET_BLOCK_SIZES=139
      PetscEnum, parameter :: MATOP_AYPX=140
      PetscEnum, parameter :: MATOP_RESIDUAL=141
      PetscEnum, parameter :: MATOP_FDCOLORING_SETUP=142
      PetscEnum, parameter :: MATOP_FIND_OFFBLOCK_ENTRIES=143
      PetscEnum, parameter :: MATOP_MPICONCATENATESEQ=144
      PetscEnum, parameter :: MATOP_DESTROYSUBMATRICES=145
      PetscEnum, parameter :: MATOP_TRANSPOSE_SOLVE=146
      PetscEnum, parameter :: MATOP_GET_VALUES_LOCAL=147
!
!
!
      PetscEnum, parameter :: MATRIX_BINARY_FORMAT_DENSE=-1
!
! MPChacoGlobalType
      PetscEnum, parameter :: MP_CHACO_MULTILEVEL_KL=0
      PetscEnum, parameter :: MP_CHACO_SPECTRAL=1
      PetscEnum, parameter :: MP_CHACO_LINEAR=2
      PetscEnum, parameter :: MP_CHACO_RANDOM=3
      PetscEnum, parameter :: MP_CHACO_SCATTERED=4
!
! MPChacoLocalType
      PetscEnum, parameter :: MP_CHACO_KERNIGHAN_LIN=0
      PetscEnum, parameter :: MP_CHACO_NONE=1
!
! MPChacoEigenType
      PetscEnum, parameter :: MP_CHACO_LANCZOS=0
      PetscEnum, parameter :: MP_CHACO_RQI_SYMMLQ=1
!
! MPPTScotchStrategyType
      PetscEnum, parameter :: MP_PTSCOTCH_QUALITY = 0
      PetscEnum, parameter :: MP_PTSCOTCH_SPEED = 1
      PetscEnum, parameter :: MP_PTSCOTCH_BALANCE = 2
      PetscEnum, parameter :: MP_PTSCOTCH_SAFETY = 3
      PetscEnum, parameter :: MP_PTSCOTCH_SCALABILITY = 4

! PetscScalarPrecision
      PetscEnum, parameter :: PETSC_SCALAR_DOUBLE=0
      PetscEnum, parameter :: PETSC_SCALAR_SINGLE=1
      PetscEnum, parameter :: PETSC_SCALAR_LONG_DOUBLE=2
!
!     CUSPARSE enumerated types
!
#if defined(PETSC_HAVE_CUDA)
      PetscEnum, parameter :: MAT_CUSPARSE_CSR=0
      PetscEnum, parameter :: MAT_CUSPARSE_ELL=1
      PetscEnum, parameter :: MAT_CUSPARSE_HYB=2
      PetscEnum, parameter :: MAT_CUSPARSE_MULT_DIAG=0
      PetscEnum, parameter :: MAT_CUSPARSE_MULT_OFFDIAG=1
      PetscEnum, parameter :: MAT_CUSPARSE_MULT=2
      PetscEnum, parameter :: MAT_CUSPARSE_ALL=3
#endif

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MAT
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MATFDCOLORING
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_MATNULLSPACE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FLUSH_ASSEMBLY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FINAL_ASSEMBLY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_NONE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_LU
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_CHOLESKY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_ILU
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_ICC
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_ILUDT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_QR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_DO_NOT_GET_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_GET_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_OPTION_MIN
!DEC$ ATTRIBUTES DLLEXPORT::MAT_UNUSED_NONZERO_LOCATION_ERR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_ROW_ORIENTED
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SYMMETRIC
!DEC$ ATTRIBUTES DLLEXPORT::MAT_STRUCTURALLY_SYMMETRIC
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FORCE_DIAGONAL_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_IGNORE_OFF_PROC_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_USE_HASH_TABLE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_KEEP_NONZERO_PATTERN
!DEC$ ATTRIBUTES DLLEXPORT::MAT_IGNORE_ZERO_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_USE_INODES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_HERMITIAN
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SYMMETRY_ETERNAL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_NEW_NONZERO_LOCATION_ERR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_IGNORE_LOWER_TRIANGULAR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_ERROR_LOWER_TRIANGULAR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_GETROW_UPPERTRIANGULAR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SPD
!DEC$ ATTRIBUTES DLLEXPORT::MAT_NO_OFF_PROC_ZERO_ROWS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_NO_OFF_PROC_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_NEW_NONZERO_LOCATIONS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_NEW_NONZERO_ALLOCATION_ERR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SUBSET_OFF_PROC_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SUBMAT_SINGLEIS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_STRUCTURE_ONLY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_OPTION_MAX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SHIFT_NONE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SHIFT_NONZERO
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SHIFT_POSITIVE_DEFINITE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SHIFT_INBLOCKS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_NOERROR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_STRUCT_ZEROPIVOT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_NUMERIC_ZEROPIVOT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_OUTMEMORY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTOR_OTHER
!DEC$ ATTRIBUTES DLLEXPORT::MAT_DO_NOT_COPY_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_COPY_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_SHARE_NONZERO_PATTERN
!DEC$ ATTRIBUTES DLLEXPORT::DIFFERENT_NONZERO_PATTERN
!DEC$ ATTRIBUTES DLLEXPORT::SUBSET_NONZERO_PATTERN
!DEC$ ATTRIBUTES DLLEXPORT::SAME_NONZERO_PATTERN
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_BLOCK_SIZE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_NZ_ALLOCATED
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_NZ_USED
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_NZ_UNNEEDED
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_MEMORY
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_ASSEMBLIES
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_MALLOCS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_FILL_RATIO_GIVEN
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_FILL_RATIO_NEEDED
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INFO_FACTOR_MALLOCS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INITIAL_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_REUSE_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_IGNORE_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_INPLACE_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_GLOBAL_MAX
!DEC$ ATTRIBUTES DLLEXPORT::MAT_GLOBAL_SUM
!DEC$ ATTRIBUTES DLLEXPORT::MAT_COMPOSITE_ADDITIVE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_COMPOSITE_MULTIPLICATIVE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_DIAGONAL_FILL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_USEDT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_DT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_DTCOL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_DTCOUNT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_FILL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_LEVELS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_PIVOT_IN_BLOCKS
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_ZERO_PIVOT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_SHIFT_TYPE
!DEC$ ATTRIBUTES DLLEXPORT::MAT_FACTORINFO_SHIFT_AMOUNT
!DEC$ ATTRIBUTES DLLEXPORT::SOR_FORWARD_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_BACKWARD_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_SYMMETRIC_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_LOCAL_FORWARD_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_LOCAL_BACKWARD_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_LOCAL_SYMMETRIC_SWEEP
!DEC$ ATTRIBUTES DLLEXPORT::SOR_ZERO_INITIAL_GUESS
!DEC$ ATTRIBUTES DLLEXPORT::SOR_EISENSTAT
!DEC$ ATTRIBUTES DLLEXPORT::SOR_APPLY_UPPER
!DEC$ ATTRIBUTES DLLEXPORT::SOR_APPLY_LOWER
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROWMATOP_RESTORE_ROW
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_ADD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_TRANSPOSE_ADD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOLVE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOLVE_ADD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOLVE_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOLVE_TRANSPOSE_ADD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_LUFACTOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CHOLESKYFACTOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GETINFO
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_EQUAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_DIAGONAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_DIAGONAL_SCALE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_NORM
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ASSEMBLY_BEGIN
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ASSEMBLY_END
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_OPTION
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ZERO_ENTRIES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ZERO_ROWS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_LUFACTOR_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_LUFACTOR_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CHOLESKY_FACTOR_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CHOLESKY_FACTOR_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SETUP_PREALLOCATION
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ILUFACTOR_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ICCFACTOR_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_DIAGONAL_BLOCK
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_DUPLICATE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FORWARD_SOLVE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_BACKWARD_SOLVE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ILUFACTOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ICCFACTOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_AXPY
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CREATE_SUBMATRICES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_INCREASE_OVERLAP
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_VALUES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_COPY
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_MAX
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SCALE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SHIFT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_DIAGONAL_SET
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ZERO_ROWS_COLUMNS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_RANDOM
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_IJ
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RESTORE_ROW_IJ
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_COLUMN_IJ
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RESTORE_COLUMN_IJ
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FDCOLORING_CREATE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_COLORING_PATCH
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_UNFACTORED
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_PERMUTE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_BLOCKED
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CREATE_SUBMATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_DESTROY
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_VIEW
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CONVERT_FROM
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MATMAT_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MATMAT_MULT_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MATMAT_MULT_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_LOCAL_TO_GLOBAL_MAP
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_ZERO_ROWS_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_MAX_ABS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_MIN_ABS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CONVERT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_COLORING
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_ADIFOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FD_COLORING_APPLY
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_FROM_OPTIONS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_CONSTRAINED
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_TRANSPOSE_CONSTRAIN
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FIND_ZERO_DIAGONALS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_MULTIPLE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SOLVE_MULTIPLE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_INERTIA
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_LOAD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_IS_SYMMETRIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_IS_HERMITIAN
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_IS_STRUCTURALLY_SYMMETRIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_BLOCKEDLOCAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CREATE_VECS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_MULT_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_MULT_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_PTAP
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_PTAP_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_PTAP_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_TRANSPOSE_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_TRANSPOSE_MULT_SYMBO
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_TRANSPOSE_MULT_NUMER
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CONJUGATE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_ROW
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_REAL_PART
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_IMAGINARY_PART
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_UPPER_TRIANGULAR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RESTORE_ROW_UPPER_TRIANG
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_SOLVE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MAT_SOLVE_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_ROW_MIN
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_COLUMN_VECTOR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MISSING_DIAGONAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_SEQ_NONZERO_STRUCTUR
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CREATE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_GHOSTS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_LOCAL_SUB_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RESTORE_LOCALSUB_MATRIX
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_DIAGONAL_BLOCK
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_HERMITIAN_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_HERMITIAN_TRANSPOSE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MULT_HERMITIAN_TRANS_ADD
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_MULTI_PROC_BLOCK
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FIND_NONZERO_ROWS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_COLUMN_NORMS
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_INVERT_BLOCK_DIAGONAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_CREATE_SUB_MATRICES_MPI
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_VALUES_BATCH
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE_MAT_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE_MAT_MULT_SYMBO
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE_MAT_MULT_NUMER
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE_COLORING_CREAT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANS_COLORING_APPLY_SPT
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANS_COLORING_APPLY_DEN
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RART
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RART_SYMBOLIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RART_NUMERIC
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_SET_BLOCK_SIZES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_AYPX
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_RESIDUAL
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_FDCOLORING_SETUP
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_MPICONCATENATESEQ
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_DESTROYSUBMATRICES
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_TRANSPOSE_SOLVE
!DEC$ ATTRIBUTES DLLEXPORT::MATOP_GET_VALUES_LOCAL
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_MULTILEVEL_KL
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_SPECTRAL
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_LINEAR
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_RANDOM
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_SCATTERED
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_KERNIGHAN_LIN
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_NONE
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_LANCZOS
!DEC$ ATTRIBUTES DLLEXPORT::MP_CHACO_RQI_SYMMLQ
!DEC$ ATTRIBUTES DLLEXPORT::MP_PTSCOTCH_QUALITY
!DEC$ ATTRIBUTES DLLEXPORT::MP_PTSCOTCH_SPEED
!DEC$ ATTRIBUTES DLLEXPORT::MP_PTSCOTCH_BALANCE
!DEC$ ATTRIBUTES DLLEXPORT::MP_PTSCOTCH_SAFETY
!DEC$ ATTRIBUTES DLLEXPORT::MP_PTSCOTCH_SCALABILITY
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SCALAR_DOUBLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SCALAR_SINGLE
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_SCALAR_LONG_DOUBLE
#if defined(PETSC_HAVE_CUDA)
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_CSR
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_ELL
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_HYB
!DEC$ ATTRIBUTES DLLEXPORT::
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_MULT_DIAG
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_MULT_OFFDIAG
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_MULT
!DEC$ ATTRIBUTES DLLEXPORT::MAT_CUSPARSE_ALL
#endif
#endif
