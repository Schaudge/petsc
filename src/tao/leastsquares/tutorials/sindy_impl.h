#ifndef _SINDYIMPL_H
#define _SINDYIMPL_H

#include <petsc/private/petscimpl.h>
#include "sindy.h"

typedef struct {
  PetscInt  B,N,dim;
  Vec       *X;
  Mat       Theta;
  Mat       *Thetas;
  PetscReal *column_scales;
  char      **names;
  char      *names_data;
  PetscInt  max_name_size;
  Variable  output_var;
} Data;

struct _p_Basis {
  PetscInt  poly_order,sine_order,cross_term_range;
  PetscBool normalize_columns,monolithic;
  Data      data;
  PetscBool monitor;
};

struct _p_SparseReg {
  PetscReal threshold;
  PetscInt  iterations;
  PetscBool monitor;
};

typedef enum {
  VECTOR = 0, SCALAR = 1
} VariableType;

struct _p_Variable {
  const char*  name;
  PetscInt     name_size;
  PetscScalar* scalar_data;
  Vec*         vec_data;
  DM           dm;
  PetscInt     N,dim,coord_dim;
  VariableType type;
  PetscInt     cross_term_dim;
  PetscInt     coord_dim_sizes[3];
  PetscInt     coord_dim_sizes_total;
  PetscInt     data_size_per_dof;
  PetscBool    own_data;
};


PETSC_EXTERN PetscBool SINDyRegisterAllCalled;
PETSC_EXTERN PetscErrorCode SINDyRegisterAll(void);
PETSC_EXTERN PetscErrorCode SINDyInitializePackage(void);
PETSC_EXTERN PetscErrorCode SINDyFinalizePackage(void);
PETSC_EXTERN PetscErrorCode VariableInitializePackage(void);
PETSC_EXTERN PetscErrorCode VariableFinalizePackage(void);
PETSC_EXTERN PetscErrorCode SparseRegInitializePackage(void);
PETSC_EXTERN PetscErrorCode SparseRegFinalizePackage(void);

PETSC_EXTERN PetscLogEvent SINDy_BasisCreate;
PETSC_EXTERN PetscLogEvent SINDy_BasisAddVariables;
PETSC_EXTERN PetscLogEvent SINDy_BasisPrint;
PETSC_EXTERN PetscLogEvent SINDy_FindSparseCoefficients;

PETSC_EXTERN PetscLogEvent Variable_Print;
PETSC_EXTERN PetscLogEvent Variable_DifferentiateSpatial;
PETSC_EXTERN PetscLogEvent Variable_ExtractDataByDim;

PETSC_EXTERN PetscLogEvent SparseReg_STLSQ;
PETSC_EXTERN PetscLogEvent SparseReg_LS;

#endif
