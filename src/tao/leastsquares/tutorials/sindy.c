#include "sindy_impl.h"

PetscClassId  SINDY_CLASSID;
PetscLogEvent SINDy_FindSparseCoefficients;
PetscLogEvent SINDy_BasisPrint;
PetscLogEvent SINDy_BasisAddVariables;
PetscLogEvent SINDy_BasisCreate;

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

PETSC_EXTERN PetscInt SINDyCountBases(PetscInt dim, PetscInt poly_order, PetscInt sine_order)
{
  return n_choose_k(dim + poly_order, poly_order) + dim *  2 * sine_order;
}

static PetscErrorCode SINDyRecordBasisFlops(PetscInt N, PetscInt dim, PetscInt poly_order, PetscInt sine_order)
{
  PetscErrorCode ierr;
  PetscInt64     flops = dim * 4 * sine_order;
  if (poly_order >= 2) flops += n_choose_k(dim + poly_order, poly_order) * ((poly_order - 1) * dim - 1) / (dim + 1) + 1;
  ierr = PetscLogFlops(N * flops);CHKERRQ(ierr);
  return(0);
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
  ierr = SINDyInitializePackage();CHKERRQ(ierr);
  PetscLogEventBegin(SINDy_BasisCreate,0,0,0,0);

  ierr = PetscMalloc1(1, &basis);CHKERRQ(ierr);
  basis->poly_order = poly_order;
  basis->sine_order = sine_order;
  basis->cross_term_range = -1;
  basis->normalize_columns = PETSC_FALSE;
  basis->monitor = PETSC_FALSE;
  basis->monolithic = PETSC_FALSE;

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
  PetscLogEventEnd(SINDy_BasisCreate,0,0,0,0);
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

PETSC_EXTERN PetscErrorCode SINDyBasisGetCrossTermRange(Basis basis, PetscInt *cross_term_range)
{
  PetscFunctionBegin;
  *cross_term_range = basis->cross_term_range;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyGetResidual(Basis basis, PetscReal* res_norm)
{
  PetscFunctionBegin;
  *res_norm = basis->data.res_norm;
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
  if (is_coord_output && is_coord_input) {
    for (PetscInt i = 0; i < 3; i++) {
      if (var->coord_dim_sizes[i] != out->coord_dim_sizes[i]) {
        SETERRQ4(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,
                "Inconsistent coordinate sizes. Coordinate dimension %d for var \"%s\" is not the same as for output variable. %d != %d.",
                i,var->coord_dim_sizes[i],out->coord_dim_sizes[i],var->name);
      }
    }
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
    ierr = VariablePrint(var);CHKERRQ(ierr);
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

      if (cross_term_range != -1 && 2*cross_term_range+1 <= var->dim) {
        d2 = (d-cross_term_range + d2 + var->dim) % var->dim;
      }
      ierr = DMDAVecGetArrayDOFRead(var->dm,var->vec_data[n],&p);CHKERRQ(ierr);
      if (var->coord_dim == 1) {
        *val = ((PetscScalar **) p)[i][d2];
      } else if (var->coord_dim == 2) {
        *val = ((PetscScalar ***) p)[j][i][d2];
      } else if (var->coord_dim == 3) {
        *val = ((PetscScalar ****) p)[k][j][i][d2];
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

/* Scale the columns to have the same norm (=1). */
static PetscErrorCode NormalizeColumns(Mat Theta, PetscScalar* column_scales)
{
  PetscErrorCode ierr;
  PetscInt       i,m,n,M,N;
  PetscReal      *data;

  PetscFunctionBegin;
  ierr = MatGetSize(Theta, &M, &N);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Theta, &data);CHKERRQ(ierr);
  for (n = 0; n < N; n++) column_scales[n] = 0;
  i = 0;
  for (n = 0; n < N; n++) {
    for (m = 0; m < M; m++) {
      column_scales[n] += data[i]*data[i];
      i++;
    }
  }
  for (n = 0; n < N; n++) column_scales[n] = PetscSqrtReal(column_scales[n]);
  i = 0;
  for (n = 0; n < N; n++) {
    for (m = 0; m < M; m++) {
      if (column_scales[n]) {
        data[i] /= column_scales[n];
      }
      i++;
    }
  }
  ierr = MatDenseRestoreArray(Theta, &data);CHKERRQ(ierr);
  PetscLogFlops(M*N + 2*M + N + M*N);
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
  PetscLogEventBegin(SINDy_BasisAddVariables,0,0,0,0);
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
      ierr = VariablePrint(vars[v]);CHKERRQ(ierr);
    }
  }

  /* Set cross term sizes. */
  basis->monolithic = (basis->cross_term_range == -1);
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
  ierr = PetscCalloc1(output_dim, &basis->data.Thetas);CHKERRQ(ierr);

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
    ierr = MatDenseRestoreArray(Theta, &Theta_data);CHKERRQ(ierr);
    if (i != out->data_size_per_dof*B) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_COR,"Computed a number basis functions (%d) different than the size of the basis matrix (%d)",
               i, out->data_size_per_dof*B);
    }
    if (basis->normalize_columns) {
      ierr = NormalizeColumns(Theta, &basis->data.column_scales[d*B]);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    if (basis->monitor) {
      PetscInt M, N;
      ierr = MatGetSize(Theta, &M, &N);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy basis matrix for dof %d: %d x %d\n", d, M, N);CHKERRQ(ierr);
      ierr = MatView(Theta, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
    if (basis->monolithic) {
      basis->data.Theta = basis->data.Thetas[d];
      basis->data.Thetas[d] = NULL;
      break;
    }
  }

  basis->data.max_name_size = 0;
  for (b = 0; b < basis->data.B; b++) {
    basis->data.max_name_size = PetscMax(strlen(basis->data.names[b]), (size_t) basis->data.max_name_size);
  }
  if (basis->poly_order >= 0) {
    ierr = PetscFree(poly_terms);CHKERRQ(ierr);
  }
  SINDyRecordBasisFlops(out->coord_dim_sizes_total * basis->data.N, cross_term_dim, basis->poly_order, basis->sine_order);
  PetscLogEventEnd(SINDy_BasisAddVariables,0,0,0,0);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeResidualNormSquared(Mat A, Vec b, Vec x, PetscReal* res_norm)
{
  PetscErrorCode ierr;
  Vec            Ax;

  PetscFunctionBegin;
  ierr = VecDuplicate(b, &Ax);CHKERRQ(ierr);
  ierr = MatMult(A, x, Ax);CHKERRQ(ierr);
  ierr = VecAXPY(Ax, -1.0, b);CHKERRQ(ierr);
  ierr = VecNorm(Ax, NORM_2, res_norm);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyFindSparseCoefficients(Basis basis, SparseReg sparse_reg, PetscInt output_dim, Vec* Xis)
{
  PetscErrorCode  ierr;
  PetscInt        d,b;
  PetscReal       *xi_data, res_norm;
  Vec             *dim_vecs;

  PetscFunctionBegin;
  PetscLogEventBegin(SINDy_FindSparseCoefficients,0,0,0,0);
  if (!basis->data.output_var) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,"Output variable must be set before calling this function");
  }
  if (output_dim != basis->data.output_var->dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,
             "the given output dim (=%d) must match the dim of the output var (=%d)", output_dim, basis->data.output_var->dim);
  }

  /* Separate out each dimension of the data. */
  ierr = VariableExtractDataByDim(basis->data.output_var, &dim_vecs);
  if (basis->monitor) {
    for (d = 0; d < output_dim; d++) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "SINDy: output variable %s[%d]:\n", basis->data.output_var->name, d);CHKERRQ(ierr);
      ierr = VecView(dim_vecs[d], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    }
  }

  /* Run regression on each dimension of the data. */
  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_SELF, "SINDy: dimensions: %d, matrix size: %d x %d\n", output_dim, basis->data.output_var->data_size_per_dof, basis->data.B);
  }
  for (d = 0; d < output_dim; d++) {
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_SELF, "SINDy: dimension: %d, starting least squares\n", d);
    }
    if (basis->monolithic) {
      ierr = SparseRegSTLSQR(sparse_reg, basis->data.Theta, dim_vecs[d], NULL, Xis[d]);CHKERRQ(ierr);
    } else {
      ierr = SparseRegSTLSQR(sparse_reg, basis->data.Thetas[d], dim_vecs[d], NULL, Xis[d]);CHKERRQ(ierr);
    }
  }

  /* Compute residual. */
  basis->data.res_norm = 0;
  for (d = 0; d < output_dim; d++) {
    if (basis->monolithic) {
      ierr = ComputeResidualNormSquared(basis->data.Theta, dim_vecs[d], Xis[d], &res_norm);CHKERRQ(ierr);
    } else {
      ierr = ComputeResidualNormSquared(basis->data.Thetas[d], dim_vecs[d], Xis[d], &res_norm);CHKERRQ(ierr);
    }
    basis->data.res_norm += res_norm;
  }
  basis->data.res_norm = PetscSqrtReal(basis->data.res_norm);

  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_WORLD, "SINDy: %s Xi\n", basis->normalize_columns ? " scaled" : "");
    ierr = SINDyBasisPrint(basis, output_dim, Xis);
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
    PetscLogFlops(output_dim * basis->data.B);
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_WORLD, "SINDy: Xi\n");
      ierr = SINDyBasisPrint(basis, output_dim, Xis);
    }
  }

  /* Destroy. */
  for (d = 0; d < output_dim; d++) {
    ierr = VecDestroy(&dim_vecs[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dim_vecs);CHKERRQ(ierr);
  PetscLogEventEnd(SINDy_FindSparseCoefficients,0,0,0,0);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisPrint(Basis basis, PetscInt output_dim, Vec* Xis)
{
  PetscErrorCode   ierr;
  const PetscReal  **xi_data;
  PetscInt         base_d, max_d, d, b;
  const PetscInt   max_columns = 8;

  PetscFunctionBegin;
  PetscLogEventBegin(SINDy_BasisPrint,0,0,0,0);
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
      else printf("   %*s[%d]", output_name_size - 2 - (int) PetscCeilReal(PetscLog10Real(d+2)), basis->data.output_var->name, d);
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
  PetscLogEventEnd(SINDy_BasisPrint,0,0,0,0);
  PetscFunctionReturn(0);
}
