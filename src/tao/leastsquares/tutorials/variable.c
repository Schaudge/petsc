#include "sindy_impl.h"

PetscClassId  VARIABLE_CLASSID;
PetscLogEvent Variable_Print;
PetscLogEvent Variable_DifferentiateSpatial;
PetscLogEvent Variable_ExtractDataByDim;

PetscErrorCode VariableCreate(const char* name, Variable* var_p)
{
  PetscErrorCode ierr;
  Variable       var;

  PetscFunctionBegin;
  PetscValidPointer(var_p,2);
  *var_p = NULL;
  ierr = VariableInitializePackage();CHKERRQ(ierr);

  ierr = PetscMalloc1(1, &var);CHKERRQ(ierr);
  var->name        = name;
  var->name_size   = strlen(name);
  var->scalar_data = NULL;
  var->vec_data    = NULL;
  var->dm          = NULL;
  var->N           = 0;
  var->dim         = 0;
  var->coord_dim   = 0;
  var->type        = VECTOR;

  *var_p = var;
  PetscFunctionReturn(0);
}

PetscErrorCode VariableDestroy(Variable* var_p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*var_p) PetscFunctionReturn(0);
  ierr = PetscFree(*var_p);CHKERRQ(ierr);
  *var_p = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode VariableSetScalarData(Variable var, PetscInt N, PetscScalar* scalar_data)
{
  PetscFunctionBegin;
  if (scalar_data && var->scalar_data) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Already has scalar data set");
  if (scalar_data && var->vec_data)    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Already set to vector data");
  var->N = N;
  var->scalar_data = scalar_data;
  var->type        = SCALAR;
  var->dim         = 1;
  var->coord_dim   = 1;
  var->coord_dim_sizes[0] = 1;
  var->coord_dim_sizes[1] = 1;
  var->coord_dim_sizes[2] = 1;
  var->data_size_per_dof = N;
  var->coord_dim_sizes_total = 1;
  PetscFunctionReturn(0);
}

PetscErrorCode VariableSetVecData(Variable var, PetscInt N, Vec* vec_data, DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vec_data && var->scalar_data) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Already has vector data set");
  if (vec_data && var->vec_data)    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Already set to scalar data");
  var->N        = N;
  var->vec_data = vec_data;
  var->dm       = dm;
  var->type     = VECTOR;
  var->coord_dim_sizes[0] = 1;
  var->coord_dim_sizes[1] = 1;
  var->coord_dim_sizes[2] = 1;
  if (var->N) {
    if (dm) {
      // ierr = DMGetDimension(dm, &var->coord_dim);CHKERRQ(ierr);
      ierr = DMGetCoordinateDim(dm, &var->coord_dim);CHKERRQ(ierr);
      if (var->coord_dim > 3) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Coordinate dimension must be <= 3 but got %d", var->coord_dim);
      }
      ierr = DMDAGetDof(dm, &var->dim);CHKERRQ(ierr);
      ierr = DMDAGetNumCells(dm, &var->coord_dim_sizes[0], &var->coord_dim_sizes[1], &var->coord_dim_sizes[2], &var->coord_dim_sizes_total);CHKERRQ(ierr);
    } else {
      var->coord_dim = 1;
      ierr = VecGetSize(vec_data[0], &var->dim);CHKERRQ(ierr);
      var->coord_dim_sizes_total = 1;
    }
  }
  var->data_size_per_dof = N * var->coord_dim_sizes_total;
  PetscFunctionReturn(0);
}

PetscErrorCode VariablePrint(Variable var)
{
  PetscErrorCode     ierr;
  PetscInt           n,i,vec_size;
  const PetscScalar  *x;

  PetscFunctionBegin;
  ierr = PetscPrintf(PETSC_COMM_SELF, "Variable Object: %s\n", var->name);CHKERRQ(ierr);
  if (var->type == VECTOR) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  type: vector\n");CHKERRQ(ierr);
  } else if (var->type == SCALAR) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "  type: scalar\n");CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_CORRUPT,"Invalid var type %d", var->type);
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "  N: %d, dim: %d, data_size_per_dof: %d\n",
                     var->N, var->dim, var->data_size_per_dof);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "  coord_dim: %d, coord_dim_sizes: (%d, %d, %d), coord_dim_sizes_total: %d\n",
                     var->coord_dim, var->coord_dim_sizes[0], var->coord_dim_sizes[1], var->coord_dim_sizes[2],
                     var->coord_dim_sizes_total);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "  cross_term_dim: %d\n", var->cross_term_dim);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "  Data:\n");CHKERRQ(ierr);
  if (var->type == VECTOR) {
    for (n = 0; n < var->N; n++) {
      ierr = VecGetArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
      ierr = VecGetSize(var->vec_data[n], &vec_size);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "  %3d:    ", n);CHKERRQ(ierr);
      for (i = 0; i < vec_size; i++) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "% -13.6g", x[i]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
    }
    if (var->dm) {
      ierr = DMView(var->dm, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  } else if (var->type == SCALAR) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "scalar\n");CHKERRQ(ierr);
    for (n = 0; n < var->N; n++) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "%3d: % -13.6g\n", n, var->scalar_data[n]);CHKERRQ(ierr);
    }
  }
  ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Take the der_order-th derivative along the coord_dim dimension of the given variable. */
PetscErrorCode VariableDifferentiateSpatial(Variable var, PetscInt coord_dim, PetscInt der_order, const char* name, Variable* out_var_p)
{
  PetscErrorCode ierr;
  Variable       der;

  PetscFunctionBegin;
  if(var->type == SCALAR) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Must be vector type to spatially differentiate");
  } else if(var->type != VECTOR) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"Invalid var type %d", var->type);
  } else if(der_order < 1 || der_order > 2) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Can only take first or second derivative");
  } else if((!var->dim && coord_dim > 0) || (var->dm && coord_dim >= var->coord_dim)) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_COR,"Invalid dimension to take derivative of. Got %d but should be < %d", coord_dim, var->dim);
  } else if((!var->dm && var->dim <= der_order) || (var->dm && var->coord_dim_sizes[coord_dim] <= der_order)) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_COR,"Need at least %d points to take a %d-th order derivative", der_order+1, der_order);
  }

  ierr = VariableCreate(name, &der);CHKERRQ(ierr);
  if (var->dm) {
    Vec            *der_vecs;
    PetscInt       c[3];
    PetscInt       n,i,j,k,d;
    PetscScalar    dx;
    c[0] = c[1] = c[2] = 0;
    ierr = VecDuplicateVecs(var->vec_data[0], var->N, &der_vecs);CHKERRQ(ierr);

    if (var->coord_dim == 2) {
      const PetscScalar ***u;
      PetscScalar       ***der_data;

      /* Get dx, assuming it is constant. */
      {
        DM         x_dm;
        Vec        gc;
        DMDACoor2d **coords;
        PetscInt   xs,ys,xm,ym;

        ierr = DMGetCoordinateDM(var->dm,&x_dm);CHKERRQ(ierr);
        ierr = DMGetCoordinatesLocal(var->dm,&gc);CHKERRQ(ierr);
        ierr = DMDAGetCorners(x_dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
        ierr = DMDAVecGetArrayRead(x_dm,gc,&coords);CHKERRQ(ierr);
        if (coord_dim == 1) dx = coords[ys+1][xs].y - coords[ys][xs].y;
        else                dx = coords[ys][xs+1].x - coords[ys][xs].x;
        ierr = DMDAVecRestoreArrayRead(x_dm,gc,&coords);CHKERRQ(ierr);
      }

      for (n = 0; n < var->N; n++) {
        ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&u);CHKERRQ(ierr);
        ierr = DMDAVecGetArrayDOF(var->dm,der_vecs[n],&der_data);CHKERRQ(ierr);
        if (coord_dim == 1) {
          /* Centered difference on inner points and zero on boundary points. // one-sided difference on boundary points. */
          if (der_order == 1) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (i = 0; i < var->coord_dim_sizes[0]; i++) {
                for (d = 0; d < var->dim; d++ ) {
                  if (j == 0) {
                    der_data[j][i][d] = 0; // (u[j+1][i][d] - u[j][i][d])/dx;
                  } else if (j == var->coord_dim_sizes[coord_dim]-1) {
                    der_data[j][i][d] = 0; // (u[j][i][d] - u[j-1][i][d])/dx;
                  } else {
                    der_data[j][i][d] = (u[j+1][i][d] - u[j-1][i][d])/(2*dx);
                  }
                }
              }
            }
          } else if (der_order == 2) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (i = 0; i < var->coord_dim_sizes[0]; i++) {
                for (d = 0; d < var->dim; d++ ) {
                  if (j == 0) {
                    der_data[j][i][d] = 0; // (u[j+2][i][d] - 2 * u[j+1][i][d]+ u[j][i][d])/(dx*dx);
                  } else if (j == var->coord_dim_sizes[coord_dim]-1) {
                    der_data[j][i][d] = 0; // (u[j][i][d] - 2 * u[j-1][i][d]+ u[j-2][i][d])/(dx*dx);
                  } else {
                    der_data[j][i][d] = (u[j+1][i][d] - 2 * u[j][i][d]+ u[j-1][i][d])/(dx*dx);
                  }
                }
              }
            }
          } else {
            SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"For 2D DMDA, unsupported der_order %d", der_order);
          }
        } else if (coord_dim == 0) {
          /* Centered difference on inner points and zero on boundary points. // one-sided difference on boundary points. */
          if (der_order == 1) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (i = 0; i < var->coord_dim_sizes[0]; i++) {
                for (d = 0; d < var->dim; d++ ) {
                  if (i == 0) {
                    der_data[j][i][d] = 0; // (u[j][i+1][d] - u[j][i][d])/dx;
                  } else if (i == var->coord_dim_sizes[coord_dim]-1) {
                    der_data[j][i][d] = 0; // (u[j][i][d] - u[j][i-1][d])/dx;
                  } else {
                    der_data[j][i][d] = (u[j][i+1][d] - u[j][i-1][d])/(2*dx);
                  }
                }
              }
            }
          } else if (der_order == 2) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (i = 0; i < var->coord_dim_sizes[0]; i++) {
                for (d = 0; d < var->dim; d++ ) {
                  if (i == 0) {
                    der_data[j][i][d] = 0; // (u[j][i+2][d] - 2 * u[j][i+1][d]+ u[j][i][d])/(dx*dx);
                  } else if (i == var->coord_dim_sizes[coord_dim]-1) {
                    der_data[j][i][d] = 0; // (u[j][i][d] - 2 * u[j][i-1][d]+ u[j][i-2][d])/(dx*dx);
                  } else {
                    der_data[j][i][d] = (u[j][i+1][d] - 2 * u[j][i][d]+ u[j][i-1][d])/(dx*dx);
                  }
                }
              }
            }
          } else {
            SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"For 2D DMDA, unsupported der_order %d", der_order);
          }
        } else {
          SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"For 2D DMDA, unsupported coord_dim %d", coord_dim);
        }
        ierr = DMDAVecRestoreArrayDOFRead(var->dm,var->vec_data[n],&u);CHKERRQ(ierr);
        ierr = DMDAVecRestoreArrayDOF(var->dm,der_vecs[n],&der_data);CHKERRQ(ierr);
      }
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Only supports 2D DMDA");
    }
    ierr = VariableSetVecData(der, var->N, der_vecs, var->dm);CHKERRQ(ierr);
  } else {
    /* Assume it is a 1D system with 1 DOF. */
    PetscInt          i;
    const PetscScalar *u;
    Vec               *der_vecs;
    PetscScalar       *der_data;
    PetscInt          n;
    const PetscReal   dx = 1;

    if (der_order == 1) {
      ierr = VecDuplicateVecs(var->vec_data[0], var->N, &der_vecs);CHKERRQ(ierr);
      for (n = 0; n < var->N; n++) {
        ierr = VecGetArrayRead(var->vec_data[n], &u);CHKERRQ(ierr);
        ierr = VecGetArray(der_vecs[n], &der_data);CHKERRQ(ierr);
        /* Centered difference on inner points. */
        for (i = 1; i < var->dim-1; i++) {
          der_data[i] = (u[i+1] - u[i-1])/(2*dx);
        }
        /* Zero out the first and last points. // One-sided difference on first and last points. */
        der_data[0] = 0; // (u[1] - u[0])/dx;
        der_data[var->dim-1] = 0; // (u[var->dim-1] - u[var->dim-2])/dx;
        ierr = VecRestoreArrayRead(var->vec_data[n], &u);CHKERRQ(ierr);
        ierr = VecRestoreArray(der_vecs[n], &der_data);CHKERRQ(ierr);
      }
      ierr = VariableSetVecData(der, var->N, der_vecs, NULL);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_COR,"Can only take first derivative for non-DMDA data");
    }
  }

  *out_var_p = der;
  PetscFunctionReturn(0);
}

PetscErrorCode VariableExtractDataByDim(Variable var, Vec** dim_vecs_p)
{
  PetscErrorCode ierr;
  PetscInt       i,d;
  Vec            *dim_vecs;
  PetscReal      *dim_data;

  PetscFunctionBegin;
  ierr = PetscMalloc1(var->dim, &dim_vecs);CHKERRQ(ierr);
  if (var->type == VECTOR) {
    if (var->dm) {
      void           *p;
      PetscInt       id,n,i,j,k;

      /* Warning: it's very important that these extract the data in the same order that SINDyBasisAddVariables does. */
      if (var->coord_dim == 1) {
        for (d = 0; d < var->dim; d++) {
          ierr = VecCreateSeq(PETSC_COMM_SELF, var->data_size_per_dof, &dim_vecs[d]);CHKERRQ(ierr);
          ierr = VecGetArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
          id = 0;
          for (i = 0; i < var->coord_dim_sizes[0]; i++) {
            for (n = 0; n < var->N; n++) {
              ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
              dim_data[id] = ((PetscScalar **) p)[i][d];
              ierr = DMDAVecRestoreArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
              id++;
            }
          }
          ierr = VecRestoreArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
        }
      } else if (var->coord_dim == 2) {
        for (d = 0; d < var->dim; d++) {
          ierr = VecCreateSeq(PETSC_COMM_SELF, var->data_size_per_dof, &dim_vecs[d]);CHKERRQ(ierr);
          ierr = VecGetArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
          id = 0;
          for (j = 0; j < var->coord_dim_sizes[1]; j++) {
            for (i = 0; i < var->coord_dim_sizes[0]; i++) {
              for (n = 0; n < var->N; n++) {
                ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
                dim_data[id] = ((PetscScalar ***) p)[j][i][d];
                ierr = DMDAVecRestoreArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
                id++;
              }
            }
          }
          ierr = VecRestoreArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
        }
      } else if (var->coord_dim == 3) {
        for (d = 0; d < var->dim; d++) {
          ierr = VecCreateSeq(PETSC_COMM_SELF, var->data_size_per_dof, &dim_vecs[d]);CHKERRQ(ierr);
          ierr = VecGetArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
          id = 0;
          for (k = 0; k < var->coord_dim_sizes[2]; k++) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (i = 0; i < var->coord_dim_sizes[0]; i++) {
                for (n = 0; n < var->N; n++) {
                  ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
                  dim_data[id] = ((PetscScalar ****) p)[k][j][i][d];
                  ierr = DMDAVecRestoreArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
                  id++;
                }
              }
            }
          }
          ierr = VecRestoreArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
        }
      } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",var->coord_dim);
    } else {
      for (d = 0; d < var->dim; d++) {
        ierr = VecCreateSeq(PETSC_COMM_SELF, var->data_size_per_dof, &dim_vecs[d]);CHKERRQ(ierr);
        ierr = VecGetArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
        for (i = 0; i < var->N; i++) {
          ierr = VecGetValues(var->vec_data[i], 1, &d, &dim_data[i]);CHKERRQ(ierr);
        }
        ierr = VecRestoreArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
      }
    }
  } else if (var->type == SCALAR) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, var->data_size_per_dof, var->scalar_data, &dim_vecs[0]);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"Invalid var type %d", var->type);
  }
  *dim_vecs_p = dim_vecs;
  PetscFunctionReturn(0);
}
