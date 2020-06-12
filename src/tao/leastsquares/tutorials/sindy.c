#include "sindy_impl.h"

static PetscInt64 n_choose_k(PetscInt64 n, PetscInt64 k)
{
  if (k > n) {
    return 0;
  }
  PetscInt64 r = 1;
  for (PetscInt64 d = 1; d <= k; ++d) {
    r *= n--;
    r /= d;
  }
  return r;
}

static PetscInt SINDyCountBases(PetscInt dim, PetscInt poly_order, PetscInt sine_order)
{
  return n_choose_k(dim + poly_order, poly_order) + dim *  2 * sine_order;
}

PETSC_EXTERN PetscErrorCode SINDyBasisDataGetSize(Basis basis, PetscInt *N, PetscInt *B)
{
  if (N) *N = basis->data.N;
  if (B) *B = basis->data.B;
  return(0);
}

PetscErrorCode SINDyBasisCreate(PetscInt poly_order, PetscInt sine_order, Basis* new_basis)
{
  PetscErrorCode  ierr;
  Basis basis;

  PetscFunctionBegin;
  PetscValidPointer(new_basis,3);
  *new_basis = NULL;

  ierr = PetscMalloc1(1, &basis);CHKERRQ(ierr);
  basis->poly_order = poly_order;
  basis->sine_order = sine_order;
  basis->cross_term_range = -1;
  basis->normalize_columns = PETSC_FALSE;
  basis->monitor = PETSC_FALSE;

  basis->data.B = 0;
  basis->data.N = -1;
  basis->data.dim = 0;
  basis->data.Theta = NULL;
  basis->data.Thetas = NULL;
  basis->data.column_scales = NULL;
  basis->data.names = NULL;
  basis->data.names_data = NULL;
  basis->data.max_name_size = 0;
  basis->data.output_var = NULL;

  *new_basis = basis;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyBasisSetNormalizeColumns(Basis basis, PetscBool normalize_columns)
{
  PetscFunctionBegin;
  basis->normalize_columns = normalize_columns;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyBasisSetCrossTermRange(Basis basis, PetscInt cross_term_range)
{
  PetscFunctionBegin;
  basis->cross_term_range = cross_term_range;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyBasisSetFromOptions(Basis basis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"sindy_","SINDy options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-poly_order","highest degree polynomial to use in basis","",basis->poly_order,&basis->poly_order,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-sine_order","highest frequency sine/cosine function to use in basis","",basis->sine_order,&basis->sine_order,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-cross_term_range","how many nearby variables to include in cross-terms (-1 to include all)","",basis->cross_term_range,&basis->cross_term_range,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-normalize_columns","scale each basis function column to have norm 1","",basis->normalize_columns,&basis->normalize_columns,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor","print out extra information","",basis->monitor,&basis->monitor,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisDestroy(Basis* basis)
{
  PetscErrorCode ierr;
  PetscInt       d,b;

  PetscFunctionBegin;
  if (!*basis) PetscFunctionReturn(0);

  ierr = MatDestroy(&((*basis)->data.Theta));CHKERRQ(ierr);
  if ((*basis)->data.Thetas) {
    if ((*basis)->data.output_var) {
      for (d = 0; d < (*basis)->data.output_var->dim; d++) {
        ierr = MatDestroy(&((*basis)->data.Thetas[d]));CHKERRQ(ierr);
      }
    } else {
      for (d = 0; d < (*basis)->data.dim; d++) {
        ierr = MatDestroy(&((*basis)->data.Thetas[d]));CHKERRQ(ierr);
      }
    }
    ierr = PetscFree((*basis)->data.Thetas);CHKERRQ(ierr);
  }
  if ((*basis)->data.output_var) {
    for (b = 0; b < (*basis)->data.B; b++) {
      ierr = PetscFree((*basis)->data.names[b]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*basis)->data.names);CHKERRQ(ierr);
  }

  ierr = PetscFree((*basis)->data.column_scales);CHKERRQ(ierr);
  ierr = PetscFree((*basis)->data.names_data);CHKERRQ(ierr);
  ierr = PetscFree((*basis)->data.names);CHKERRQ(ierr);
  ierr = PetscFree(*basis);CHKERRQ(ierr);
  *basis = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDySparseRegCreate(SparseReg* new_sparse_reg)
{
  PetscErrorCode  ierr;
  SparseReg sparse_reg;

  PetscFunctionBegin;
  PetscValidPointer(new_sparse_reg,3);
  *new_sparse_reg = NULL;

  ierr = PetscMalloc1(1, &sparse_reg);CHKERRQ(ierr);
  sparse_reg->threshold = 1e-5;
  sparse_reg->iterations = 10;
  sparse_reg->monitor = PETSC_FALSE;

  *new_sparse_reg = sparse_reg;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDySparseRegDestroy(SparseReg* sparse_reg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sparse_reg) PetscFunctionReturn(0);
  ierr = PetscFree(*sparse_reg);CHKERRQ(ierr);
  *sparse_reg = NULL;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDySparseRegSetThreshold(SparseReg sparse_reg, PetscReal threshold)
{
  PetscFunctionBegin;
  sparse_reg->threshold = threshold;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDySparseRegSetMonitor(SparseReg sparse_reg, PetscBool monitor)
{
  PetscFunctionBegin;
  sparse_reg->monitor = monitor;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDySparseRegSetFromOptions(SparseReg sparse_reg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"sparse_reg_","Sparse regression options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-threshold","coefficients below this are set to 0","",sparse_reg->threshold,&sparse_reg->threshold,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations","number of thresholded iterations to do","",sparse_reg->iterations,&sparse_reg->iterations,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor","print out additional information","",sparse_reg->monitor,&sparse_reg->monitor,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef struct {
  Mat       A,D;
  Vec       b;
  PetscInt  M,N,K;
} LeastSquaresCtx;

PetscErrorCode InitializeLeastSquaresData(LeastSquaresCtx *ctx, Mat A, Mat D, Vec b)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ctx->A = A;
  ctx->D = D;
  ctx->b = b;
  ierr = MatGetSize(ctx->A,&ctx->M,&ctx->N);CHKERRQ(ierr);
  if (D) {
    ierr = MatGetSize(ctx->D,&ctx->K,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  LeastSquaresCtx *ctx = (LeastSquaresCtx *)ptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatMult(ctx->A, X, F);CHKERRQ(ierr);
  ierr = VecAXPY(F, -1.0, ctx->b);CHKERRQ(ierr);
  PetscLogFlops(ctx->M*ctx->N*2);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
  /* The Jacobian is constant for this problem. */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode FormStartingPoint(Vec X)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(X,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Find x to minimize ||Ax - b||_2 + ||Dx||_1 where A is an m x n matrix. If D is
   null, it will default to the identity. The given mask tells which entries in
   x to leave out of the optimization. If null, all entries will be optimized. */
PetscErrorCode SINDySparseLeastSquares(Mat A, Vec b, Mat D, Vec x)
{
  PetscErrorCode  ierr;
  Vec             f;               /* solution, function f(x) = A*x-b */
  Mat             J;               /* Jacobian matrix, Transform matrix */
  Tao             tao;                /* Tao solver context */
  LeastSquaresCtx ctx;
  PetscBool       flg;

  PetscFunctionBegin;
  ierr = InitializeLeastSquaresData(&ctx, A, D, b);CHKERRQ(ierr);
  J = A;

  /* Allocate vector function vector */
  ierr = VecDuplicate(b, &f);CHKERRQ(ierr);

  /* Use l1dict by default. */
  ierr = PetscOptionsHasName(NULL, NULL, "-tao_brgn_regularization_type", &flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsSetValue(NULL, "-tao_brgn_regularization_type", "l1dict");CHKERRQ(ierr);
  }

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBRGN);CHKERRQ(ierr);

  /* Set initial guess */
  ierr = FormStartingPoint(x);CHKERRQ(ierr);

  /* Bind x to tao->solution. */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);

  /* Bind D to tao->data->D */
  ierr = TaoBRGNSetDictionaryMatrix(tao,D);CHKERRQ(ierr);

  /* Set the function and Jacobian routines. */
  ierr = TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&ctx);CHKERRQ(ierr);
  ierr = TaoSetJacobianResidualRoutine(tao,J,J,EvaluateJacobian,(void*)&ctx);CHKERRQ(ierr);

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyVariableCreate(const char* name, Variable* var_p)
{
  PetscErrorCode ierr;
  Variable       var;

  PetscFunctionBegin;
  PetscValidPointer(var_p,2);
  *var_p = NULL;

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

PetscErrorCode SINDyVariableDestroy(Variable* var_p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*var_p) PetscFunctionReturn(0);
  ierr = PetscFree(*var_p);CHKERRQ(ierr);
  *var_p = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyVariableSetScalarData(Variable var, PetscInt N, PetscScalar* scalar_data)
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

PetscErrorCode SINDyVariableSetVecData(Variable var, PetscInt N, Vec* vec_data, DM dm)
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

PetscErrorCode SINDyVariablePrint(Variable var)
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
        ierr = PetscPrintf(PETSC_COMM_SELF, "% -10.6g", x[i]);CHKERRQ(ierr);
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
      ierr = PetscPrintf(PETSC_COMM_SELF, "%3d: % -10.6g\n", n, var->scalar_data[n]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisValidateVariable(Basis basis, Variable var)
{
  PetscFunctionBegin;
  Variable out = basis->data.output_var;
  if (!out) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Output variable must be set before calling this function");
  }
  /* Validate variables size. */
  if (var->N != basis->data.N) {
    SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,
            "Inconsistent data size. Expected var \"%s\" to have size %d but found %d",
            var->name, basis->data.N, var->N);
  }

  if (var->coord_dim < 1 || var->coord_dim > 3) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,
            "var \"%s\" coord_dim must 1, 2, or 3, but it is %d\n",
            var->name, var->coord_dim);
  }

  /* Validate dm. */
  /* The point is that if the output variable is defined on a coordinate space, then
     every input that depends on a coordinate space needs to depend on the same
     coordinate space. */
  PetscBool is_coord_output = (out->type == VECTOR && out->dm);
  PetscBool is_coord_input = (var->type == VECTOR && var->dm);
  if (is_coord_output && is_coord_input && var->dm != out->dm) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,
            "Inconsistent coordinates. DM for var \"%s\" is not the same as for output variable.",var->name);
  }
  /* If the output variable isn't defined on a coordinate space, then for now,
     no inputs should depend on a coordinate space. */
  if (!is_coord_output && is_coord_input) {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,
            "Inconsistent coordinates. Output doesn't depend on coordinates but var \"%s\" does.",var->name);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyBasisSetOutputVariable(Basis basis, Variable var) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (basis->data.N == -1) basis->data.N = var->N;
  basis->data.output_var = var;
  ierr = SINDyBasisValidateVariable(basis, var);CHKERRQ(ierr);
  if (basis->monitor) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy output variable:\n");CHKERRQ(ierr);
    ierr = SINDyVariablePrint(var);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisGetLocalDOF(PetscInt d, PetscInt num_vars, Variable* vars, PetscInt* d_p, Variable* var_p)
{
  PetscInt v;

  PetscFunctionBegin;
  for (v = 0; v < num_vars; v++) {
    if (d - vars[v]->cross_term_dim < 0) break;
    d -= vars[v]->cross_term_dim;
  }
  if (v >= num_vars || d < 0) SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Invalid result dof=%d, v=%d", d, v);
  *d_p = d;
  *var_p = vars[v];
  PetscFunctionReturn(0);
}

/* Extracts DOF d2 from the list of variables for data point n, as related to DOF
   d at coordinate coords in the output variable.*/
static PetscErrorCode SINDyVariableGetDOF(PetscInt d, PetscInt* coords, PetscInt cross_term_range,
                                          PetscInt num_vars, Variable* vars,
                                          PetscInt n, PetscInt d2, PetscScalar *val)
{
  PetscErrorCode  ierr;
  Variable        var;

  PetscFunctionBegin;
  ierr = SINDyBasisGetLocalDOF(d2, num_vars, vars, &d2, &var);CHKERRQ(ierr);

  /* Need to extract local DOF d2 from variable var. */
  if(var->type == VECTOR) {
    if (var->dm) {
      void           *p;
      PetscInt       i,j,k;
      i = coords[0]; j = coords[1]; k = coords[2];

      ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
      if (var->coord_dim == 1) {
        *val = ((PetscScalar **) p)[i][d];
      } else if (var->coord_dim == 2) {
        *val = ((PetscScalar ***) p)[j][i][d];
      } else if (var->coord_dim == 3) {
        *val = ((PetscScalar ****) p)[k][j][i][d];
      } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"DMDA dimension not 1, 2, or 3, it is %D\n",var->coord_dim);
      ierr = DMDAVecRestoreArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
    } else {
      const PetscReal *x;
      ierr = VecGetArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
      if (cross_term_range != -1 && 2*cross_term_range+1 <= var->dim) {
        d2 = (d-cross_term_range + d2 + var->dim) % var->dim;
      }
      *val = x[d2];
      ierr = VecRestoreArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
    }
  } else if(var->type == SCALAR) {
    *val = var->scalar_data[n];
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"Invalid var type %d", var->type);
  }
  PetscFunctionReturn(0);
}

/* Generate name using the given polynomial. */
static PetscErrorCode SINDyBasisGenerateNamePolynomial(Basis basis, PetscInt num_vars, Variable* vars, const PetscInt* poly_terms, char** name_p)
{
  PetscErrorCode ierr;
  PetscInt       s, name_size, written;
  PetscInt       o, d;
  Variable       var;
  char           *name;

  PetscFunctionBegin;
  /* Count how big to make the string. */
  name_size = 0;
  for (o = basis->poly_order; o >= 0 ; o--) {
    d = poly_terms[o] - 1;
    if (d >= 0) {
      ierr = SINDyBasisGetLocalDOF(d, num_vars, vars, &d, &var);CHKERRQ(ierr);
      name_size += var->name_size + 3 + (int) PetscCeilReal(PetscLog10Real(var->dim+2));
      if (var->cross_term_dim != var->dim) {
        d = d + basis->cross_term_range;
        if (d) name_size += 4 + (int) PetscCeilReal(PetscLog10Real(d+2));
      }
    }
  }
  if (name_size == 0) name_size++; /* Constant '1'.  */
  name_size++;                     /* Null character */

  /* Build the string. */
  ierr = PetscMalloc1(name_size, &name);CHKERRQ(ierr);
  s = 0;
  for (o = basis->poly_order; o >= 0 ; o--) {
    d = poly_terms[o] - 1;
    if (d >= 0) {
      ierr = SINDyBasisGetLocalDOF(d, num_vars, vars, &d, &var);CHKERRQ(ierr);
      if (var->dim == 1) {
        written = sprintf(&name[s], "%s*", var->name);
      } else if (var->cross_term_dim == var->dim) {
        written = sprintf(&name[s], "%s[%d]*", var->name, d);
      } else {
        d = d - basis->cross_term_range;
        if (d) written = sprintf(&name[s], "%s[i%+d]*", var->name, d);
        else   written = sprintf(&name[s], "%s[i]*", var->name);
      }
      if (written < 0) SETERRQ(PETSC_COMM_SELF, 1, "sprintf error");
      s += written;
    }
  }
  /* Set to 1 or remove extra asterisk. */
  if (s == 0) strcpy(name, "1");
  else name[s-1] = '\0';

  *name_p = name;
  PetscFunctionReturn(0);
}

/* Generate name using the given sine order. */
static PetscErrorCode SINDyBasisGenerateNameSine(Basis basis, PetscInt num_vars, Variable* vars,
                                                 PetscInt order, PetscInt d, char** sin_p, char** cos_p)
{
  PetscErrorCode ierr;
  PetscInt       name_size;
  Variable       var;
  char           *sin_name, *cos_name;

  PetscFunctionBegin;
  ierr = SINDyBasisGetLocalDOF(d, num_vars, vars, &d, &var);CHKERRQ(ierr);
  name_size = 7 + var->name_size + (int) PetscCeilReal(PetscLog10Real(order+2));
  if (var->dim == 1) {
    ierr = PetscMalloc1(name_size, &sin_name);CHKERRQ(ierr);
    ierr = PetscMalloc1(name_size, &cos_name);CHKERRQ(ierr);
    if (sprintf(sin_name, "sin(%d*%s)", order, var->name) < 0) PetscFunctionReturn(1);
    if (sprintf(cos_name, "cos(%d*%s)", order, var->name) < 0) PetscFunctionReturn(1);
  } else if (var->cross_term_dim == var->dim) {
    name_size += 2 + (int) PetscCeilReal(PetscLog10Real(var->dim+2));
    ierr = PetscMalloc1(name_size, &sin_name);CHKERRQ(ierr);
    ierr = PetscMalloc1(name_size, &cos_name);CHKERRQ(ierr);
    if (sprintf(sin_name, "sin(%d*%s[%d])", order, var->name, d) < 0) PetscFunctionReturn(1);
    if (sprintf(cos_name, "cos(%d*%s[%d])", order, var->name, d) < 0) PetscFunctionReturn(1);
  } else {
    d = d - basis->cross_term_range;
    if (d) {
      name_size += 4 + (int) PetscCeilReal(PetscLog10Real(PetscAbsInt(d)+2));
      ierr = PetscMalloc1(name_size, &sin_name);CHKERRQ(ierr);
      ierr = PetscMalloc1(name_size, &cos_name);CHKERRQ(ierr);
      if (sprintf(sin_name, "sin(%d*%s[i%+d])", order, var->name, d) < 0) PetscFunctionReturn(1);
      if (sprintf(cos_name, "cos(%d*%s[i%+d])", order, var->name, d) < 0) PetscFunctionReturn(1);
    }
    else {
      name_size += 3;
      ierr = PetscMalloc1(name_size, &sin_name);CHKERRQ(ierr);
      ierr = PetscMalloc1(name_size, &cos_name);CHKERRQ(ierr);
      if (sprintf(sin_name, "sin(%d*%s[i])", order, var->name) < 0) PetscFunctionReturn(1);
      if (sprintf(cos_name, "cos(%d*%s[i])", order, var->name) < 0) PetscFunctionReturn(1);
    }
  }
  *sin_p = sin_name;
  *cos_p = cos_name;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisAddVariables(Basis basis, PetscInt num_vars, Variable* vars)
{
  PetscErrorCode  ierr;
  PetscInt        output_dim,input_dim,cross_term_dim,B;
  PetscInt        v, n, i, o, d, d2, b, c;
  PetscReal       *Theta_data;
  PetscInt        *poly_terms;
  Mat             Theta;
  PetscScalar     val;
  PetscInt        coords[3];
  Variable        out;

  PetscFunctionBegin;
  /* Validate variables size. */
  if (!num_vars) PetscFunctionReturn(0);
  if (!basis->data.output_var) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Output variable must be set before calling this function");
  }
  if (basis->data.N == -1) basis->data.N = vars[0]->N;
  for (v = 0; v < num_vars; v++) {
    ierr = SINDyBasisValidateVariable(basis, vars[v]);CHKERRQ(ierr);
  }

  if (basis->monitor) {
    for (v = 0; v < num_vars; v++) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy variable %d:\n", v);CHKERRQ(ierr);
      ierr = SINDyVariablePrint(vars[v]);CHKERRQ(ierr);
    }
  }

  /* Set cross term sizes. */
  for (v = 0; v < num_vars; v++) {
    if (basis->cross_term_range == -1) vars[v]->cross_term_dim = vars[v]->dim;
    else                               vars[v]->cross_term_dim = PetscMin(2*basis->cross_term_range+1, vars[v]->dim);
  }

  /* Count input dimensions. */
  input_dim = 0;
  for (v = 0; v < num_vars; v++) {
    input_dim += vars[v]->dim;
  }

  /* Count how many basis columns will be created. */
  cross_term_dim = 0;
  for (v = 0; v < num_vars; v++) {
    cross_term_dim += vars[v]->cross_term_dim;
  }
  basis->data.B = SINDyCountBases(cross_term_dim, basis->poly_order, basis->sine_order);

  /* Nice short names. */
  out = basis->data.output_var;
  output_dim = out->dim;
  B = basis->data.B;

  /* Allocate basis data. */
  if (basis->poly_order >= 0) {
    ierr = PetscCalloc1(basis->poly_order+1, &poly_terms);CHKERRQ(ierr);
  }
  if (basis->normalize_columns) {
    ierr = PetscCalloc1(B * output_dim, &basis->data.column_scales);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(output_dim, &basis->data.Thetas);CHKERRQ(ierr);

  /* Allocate name data. */
  ierr = PetscMalloc1(B, &basis->data.names);CHKERRQ(ierr);

  /* TODO: For now I will make the coefficients constant across all coordinates. In
     the future, we can add an option to allow the coefficients to vary across
     space. This will involve increasing the output_dim to the number of points
     times the number of outputs per point.*/
  for (d = 0; d < output_dim; d++) {
    /* Create a separate matrix for each output degree of freedom d. */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, out->data_size_per_dof, B, NULL, &Theta);CHKERRQ(ierr);
    basis->data.Thetas[d] = Theta;
    ierr = MatDenseGetArray(Theta, &Theta_data);CHKERRQ(ierr);

    i = 0;
    b = 0;
    /* Iterate through all basis functions of every order up to poly_order. */
    /* Result polynomial is product of dimensions poly_terms[o]-1 (if it's not -1). */
    for (o = 0; o <= basis->poly_order; o++) {
      poly_terms[o] = 0;
    }
    while (poly_terms[basis->poly_order] < 1) {
      /* Generate the polynomial corresponding to the powers in poly_terms. */

      /* Generate name. */
      if (d == 0) {
        ierr = SINDyBasisGenerateNamePolynomial(basis, num_vars, vars, poly_terms, &basis->data.names[b]);CHKERRQ(ierr);
        b++;
      }

      /* Compute values. */
      /* Loop through all output coordinates. */
      coords[0] = -1;
      coords[1] = coords[2] = 0;
      for (c = 0; c < out->coord_dim_sizes_total; c++) {
        /* Add one to the coordinate. */
        coords[0]++;
        for (PetscInt c = 0; c < out->coord_dim-1; c++) {
          if (coords[c] >= out->coord_dim_sizes[c]) {
            coords[c] = 0;
            coords[c+1]++;
          } else {
            break;
          }
        }
        // printf("c: %d  (%d, %d, %d)\n", c, coords[0], coords[1], coords[2]);
        if (coords[out->coord_dim-1] > out->coord_dim_sizes[out->coord_dim-1]) {
          // printf("Finished looping through coordinates\n");
          break;
        }
        for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
          Theta_data[i] = 1;
          for (o = basis->poly_order; o >= 0 ; o--) {
            d2 = poly_terms[o] - 1;
            if (d2 >= 0) {
              ierr = SINDyVariableGetDOF(d, coords, basis->cross_term_range, num_vars, vars, n, d2, &val);CHKERRQ(ierr);
              // printf("    c: %d  (%d, %d, %d), d: %d, d2: %d, n: %d, %g\n", c, coords[0], coords[1], coords[2], d, d2, n, val);
              Theta_data[i] *= val;
            }
          }
          i++;
        }
      }

      /* Add one to the poly_terms data, with carrying. */
      poly_terms[0]++;
      for (o = 0; o < basis->poly_order; o++) {
        if (poly_terms[o] > cross_term_dim) {
          poly_terms[o+1]++;
          for (PetscInt o2 = o; o2 >= 0; o2--) poly_terms[o2] = poly_terms[o+1];
        } else {
          break;
        }
      }
    }

    /* Add trig functions. */
    for (d2 = 0; d2 < cross_term_dim; d2++) {
      for (o = 1; o <= basis->sine_order; o++) {
        if (d == 0) {
          ierr = SINDyBasisGenerateNameSine(basis, num_vars, vars, o, d2, &basis->data.names[b], &basis->data.names[b+1]);CHKERRQ(ierr);
          b += 2;
        }
        /* Loop through all output coordinates. */
        coords[0] = -1;
        coords[1] = coords[2] = 0;
        for (c = 0; c < out->coord_dim_sizes_total; c++) {
          /* Add one to the coordinate. */
          coords[0]++;
          for (PetscInt c = 0; c < out->coord_dim-1; c++) {
            if (coords[c] >= out->coord_dim_sizes[c]) {
              coords[c] = 0;
              coords[c+1]++;
            } else {
              break;
            }
          }
          if (coords[out->coord_dim-1] > out->coord_dim_sizes[out->coord_dim-1]) {
            break;
          }
          for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
            ierr = SINDyVariableGetDOF(d, coords, basis->cross_term_range, num_vars, vars, n, d2, &val);CHKERRQ(ierr);
            Theta_data[i] = PetscSinReal(o * val);
            i++;
          }
        }
        /* Loop through all output coordinates. */
        coords[0] = -1;
        coords[1] = coords[2] = 0;
        for (c = 0; c < out->coord_dim_sizes_total; c++) {
          /* Add one to the coordinate. */
          coords[0]++;
          for (PetscInt c = 0; c < out->coord_dim-1; c++) {
            if (coords[c] >= out->coord_dim_sizes[c]) {
              coords[c] = 0;
              coords[c+1]++;
            } else {
              break;
            }
          }
          if (coords[out->coord_dim-1] > out->coord_dim_sizes[out->coord_dim-1]) {
            break;
          }
          for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
            ierr = SINDyVariableGetDOF(d, coords, basis->cross_term_range, num_vars, vars, n, d2, &val);CHKERRQ(ierr);
            Theta_data[i] = PetscCosReal(o * val);
            i++;
          }
        }
      }
    }
    if (i != out->data_size_per_dof*B) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"Computed a number basis functions (%d) different than the size of the basis matrix (%d)",
               i, out->data_size_per_dof*B);
    }

    if (basis->normalize_columns) {
      /* Scale the columns to have the same norm. */
      i = 0;
      for (b = 0; b < B; b++) {
        for (n = 0; n < basis->data.N; n++) {
          basis->data.column_scales[b+d*B] += Theta_data[i]*Theta_data[i];
          i++;
        }
      }
      for (b = 0; b < B; b++) {
        basis->data.column_scales[b+d*B] = PetscSqrtReal(basis->data.column_scales[b+d*B]);
      }
      i = 0;
      for (b = 0; b < B; b++) {
        for (n = 0; n < basis->data.N; n++) {
          if (basis->data.column_scales[b+d*B]) {
            Theta_data[i] /= basis->data.column_scales[b+d*B];
          }
          i++;
        }
      }
    }
    ierr = MatDenseRestoreArray(Theta, &Theta_data);CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    if (basis->monitor) {
      PetscInt M, N;
      ierr = MatGetSize(Theta, &M, &N);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy basis matrix for dof %d: %d x %d\n", d, M, N);CHKERRQ(ierr);
      ierr = MatView(Theta, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  }

  basis->data.max_name_size = 0;
  for (b = 0; b < basis->data.B; b++) {
    basis->data.max_name_size = PetscMax(strlen(basis->data.names[b]), (size_t) basis->data.max_name_size);
  }
  if (basis->poly_order >= 0) {
    ierr = PetscFree(poly_terms);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyVariableExtractDataByDim(Variable var, Vec** dim_vecs_p)
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
          for (i = 0; i < var->coord_dim_sizes[0]; i++) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
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
          for (i = 0; i < var->coord_dim_sizes[0]; i++) {
            for (j = 0; j < var->coord_dim_sizes[1]; j++) {
              for (k = 0; k < var->coord_dim_sizes[2]; k++) {
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


PetscErrorCode SINDyFindSparseCoefficientsVariable(Basis basis, SparseReg sparse_reg, PetscInt output_dim, Vec* Xis)
{
  PetscErrorCode  ierr;
  PetscInt        d,b;
  PetscReal       *xi_data;
  Vec             *dim_vecs;

  PetscFunctionBegin;
  if (!basis->data.output_var) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Output variable must be set before calling this function");
  }
  if (output_dim != basis->data.output_var->dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,
             "the given output dim (=%d) must match the dim of the output var (=%d)", output_dim, basis->data.output_var->dim);
  }

  /* Separate out each dimension of the data. */
  ierr = SINDyVariableExtractDataByDim(basis->data.output_var, &dim_vecs);
  if (basis->monitor) {
    for (d = 0; d < output_dim; d++) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy: output variable %s[%d]:\n", basis->data.output_var->name, d);CHKERRQ(ierr);
      ierr = VecView(dim_vecs[d], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  }

  basis->monolithic = PETSC_FALSE;

  /* Run regression on each dimension of the data. */
  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_SELF, "SINDy: dimensions: %d, matrix size: %d x %d\n", output_dim, basis->data.output_var->data_size_per_dof, basis->data.B);
  }
  for (d = 0; d < output_dim; d++) {
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_SELF, "SINDy: dimension: %d, starting least squares\n", d);
    }
    if (basis->monolithic) {
      ierr = SINDySequentialThresholdedLeastSquares(sparse_reg, basis->data.Theta, dim_vecs[d], NULL, Xis[d]);CHKERRQ(ierr);
    } else {
      ierr = SINDySequentialThresholdedLeastSquares(sparse_reg, basis->data.Thetas[d], dim_vecs[d], NULL, Xis[d]);CHKERRQ(ierr);
    }
  }

  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_WORLD, "SINDy: %s Xi\n", basis->normalize_columns ? " scaled" : "");
    ierr = SINDyBasisPrintVariable(basis, output_dim, Xis);
  }
  if (basis->normalize_columns) {
    /* Scale back Xi to the original values. */
    if (basis->monolithic) {
      for (d = 0; d < output_dim; d++) {
          ierr = VecGetArray(Xis[d], &xi_data);CHKERRQ(ierr);
          for (b = 0; b < basis->data.B; b++) {
            if (basis->data.column_scales[b]) {
              xi_data[b] /= basis->data.column_scales[b];
            }
          }
          ierr = VecRestoreArray(Xis[d], &xi_data);CHKERRQ(ierr);
      }
    } else {
      for (d = 0; d < output_dim; d++) {
          ierr = VecGetArray(Xis[d], &xi_data);CHKERRQ(ierr);
          for (b = 0; b < basis->data.B; b++) {
            if (basis->data.column_scales[b+d*basis->data.B]) {
              xi_data[b] /= basis->data.column_scales[b + d*basis->data.B];
            }
          }
          ierr = VecRestoreArray(Xis[d], &xi_data);CHKERRQ(ierr);
      }
    }
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_WORLD, "SINDy: Xi\n");
      ierr = SINDyBasisPrintVariable(basis, output_dim, Xis);
    }
  }

  /* Destroy. */
  for (d = 0; d < output_dim; d++) {
    ierr = VecDestroy(&dim_vecs[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dim_vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDySequentialThresholdedLeastSquares(SparseReg sparse_reg, Mat A, Vec b, Mat D, Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i,k,j,R,C;
  PetscInt       old_num_thresholded,num_thresholded;
  PetscBool      *mask;
  PetscReal      *x,*zeros;
  Mat            A_thresh;
  PetscInt       *idR, *idC_thresh;

  PetscFunctionBegin;
  ierr = SINDySparseLeastSquares(A, b, D, X);CHKERRQ(ierr);

  if (sparse_reg->threshold <= 0) PetscFunctionReturn(0);

  /* Create a workspace for thresholding. */
  ierr = MatDuplicate(A, MAT_COPY_VALUES, &A_thresh);CHKERRQ(ierr);

  ierr = MatGetSize(A, &R, &C);CHKERRQ(ierr);
  ierr = PetscCalloc1(R, &zeros);CHKERRQ(ierr);
  ierr = PetscMalloc3(C, &mask, R, &idR, C, &idC_thresh);CHKERRQ(ierr);
  for (i=0;i<R;i++) idR[i] = i;
  for (i=0;i<C;i++) mask[i] = PETSC_FALSE;

  /* Repeatedly threshold and perform least squares on the non-thresholded values. */
  num_thresholded = 0;
  ierr = MatCopy(A, A_thresh, SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  for (k = 0; k < sparse_reg->iterations; k++) {
    /* Threshold the data. */
    old_num_thresholded = num_thresholded;
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    for (j = 0; j < C; j++) {
      if (!mask[j]) {
        if (PetscAbsReal(x[j]) < sparse_reg->threshold) {
          x[j] = 0;
          idC_thresh[num_thresholded] = j;
          num_thresholded++;
          mask[j] = PETSC_TRUE;
        }
      } else {
          x[j] = 0;
      }
    }
    ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_SELF, "SparseReg: iteration: %d, nonzeros: %d\n", k, C - num_thresholded);
    }
    if (old_num_thresholded == num_thresholded) break;

    /* Zero out those columns of the matrix. */
    for (j = old_num_thresholded; j < num_thresholded; j++) {
      ierr = MatSetValues(A_thresh,R,idR,1,idC_thresh+j,zeros,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A_thresh,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Run sparse least squares on the non-zero basis functions. */
    ierr = SINDySparseLeastSquares(A_thresh, b, D, X);CHKERRQ(ierr);
  }

  /* Maybe I should zero out the thresholded entries again here, just to make sure Tao didn't mess them up. */
  ierr = VecGetArray(X, &x);CHKERRQ(ierr);
  for (j = 0; j < num_thresholded; j++) {
    x[idC_thresh[j]] = 0;
  }
  ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  ierr = PetscFree(zeros);CHKERRQ(ierr);
  ierr = PetscFree3(mask, idR, idC_thresh);CHKERRQ(ierr);
  ierr = MatDestroy(&A_thresh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode SINDyBasisPrintVariable(Basis basis, PetscInt output_dim, Vec* Xis)
{
  PetscErrorCode   ierr;
  const PetscReal  **xi_data;
  PetscInt         base_d, max_d, d, b;
  const PetscInt   max_columns = 8;

  PetscFunctionBegin;
  if (!basis->data.output_var) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Output variable must be set before calling this function");
  }
  if (output_dim != basis->data.output_var->dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,
             "the given output dim (=%d) must match the dim of the output var (=%d)", output_dim, basis->data.output_var->dim);
  }


  /* Get Xi data. */
  ierr = PetscMalloc1(output_dim, &xi_data);CHKERRQ(ierr);
  for (d = 0; d < output_dim; d++) {
    ierr = VecGetArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }

  PetscInt output_name_size;
  output_name_size = strlen(basis->data.output_var->name);
  if (output_dim >= 1) output_name_size += 2 + PetscCeilReal(PetscLog10Real(output_dim+2));
  output_name_size = PetscMax(output_name_size, 11);

  /* Print Xi data. */
  for (base_d = 0; base_d < output_dim; base_d += max_columns) {
    max_d = PetscMin(base_d + max_columns, output_dim);

    /* Print header line. */
    printf(" %*s", basis->data.max_name_size+1, "");
    for (d = base_d; d < max_d; d++) {
      if (output_dim == 1) printf("   %*s", output_name_size, basis->data.output_var->name);
      else printf("   %*s[%d]", output_name_size - 2 - (int) PetscCeilReal(PetscLog10Real(output_dim+2)), basis->data.output_var->name, d);
    }
    printf("\n");

    /* Print results. */
    for (b = 0; b < basis->data.B; b++) {
      printf(" %-*s ", basis->data.max_name_size+1, basis->data.names[b]);
      for (d = base_d; d < max_d; d++) {
        if (xi_data[d][b] == 0) {
          printf("   %*s", output_name_size, "0");
        } else {
          printf("   % *.4e", output_name_size, xi_data[d][b]);
        }
      }
      printf("\n");
    }
    if (max_d != output_dim) {
      printf("\n");
    }
  }

  /* Restore Xi data. */
  for (d = 0; d < output_dim; d++) {
    ierr = VecRestoreArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(xi_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
