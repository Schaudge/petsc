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
    PetscInt     N,dim;
    VariableType type;
    PetscInt     cross_term_dim;
};

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

PetscErrorCode SINDyInitializeBasesNames(Basis basis) {
  PetscErrorCode   ierr;
  PetscInt         i, b, o, d, names_size, basis_dim;
  PetscInt         *names_offsets;
  PetscInt         *poly_terms;

  PetscFunctionBegin;
  if (basis->data.names) PetscFunctionReturn(0);

  if (basis->poly_order >= 0) {
    ierr = PetscCalloc1(basis->poly_order+1, &poly_terms);CHKERRQ(ierr);
  }

  /* Count string size. */
  ierr = PetscMalloc1(basis->data.B, &basis->data.names);CHKERRQ(ierr);
  ierr = PetscMalloc1(basis->data.B, &names_offsets);CHKERRQ(ierr);

  names_size = 0;
  b = 0;
  if (basis->monolithic) {
    basis_dim = basis->data.dim;
  } else {
    basis_dim = 2*basis->cross_term_range+1;
  }
  while (poly_terms[basis->poly_order] < 1) {
    names_offsets[b] = names_size;
    b++;
    i = 0;
    for (o = basis->poly_order; o >= 0 ; o--) {
      d = poly_terms[o] - 1;
      if (d >= 0) {
        i++;
      }
    }
    if (i == 0) {
      i++;
    }
    i++;
    names_size += i+1;

    /* Add one to the poly_terms data, with carrying. */
    poly_terms[0]++;
    for (o = 0; o < basis->poly_order; o++) {
      if (poly_terms[o] > basis_dim) {
        poly_terms[o+1]++;
        for (PetscInt o2 = o; o2 >= 0; o2--) poly_terms[o2] = poly_terms[o+1];
      } else {
        break;
      }
    }
  }
  for (d = 0; d < basis_dim; d++) {    /* For each nearby degree of freedom d. */
    /* Add basis functions using this degree of freedom. */
    for (o = 1; o <= basis->sine_order; o++) {
      names_offsets[b] = names_size;
      b++;
      names_size += 4 + PetscCeilReal(PetscLog10Real(o+2)) + 3;
      /* Null character */
      names_size += 1;
      names_offsets[b] = names_size;
      b++;
      names_size += 4 + PetscCeilReal(PetscLog10Real(o+2)) + 3;
      /* Null character */
      names_size += 1;
    }
  }

  /* Build string. */
  ierr = PetscMalloc1(names_size, &basis->data.names_data);CHKERRQ(ierr);
  b = 0;
  for (o = 0; o <= basis->poly_order; o++) {
    poly_terms[o] = 0;
  }
  while (poly_terms[basis->poly_order] < 1) {
    basis->data.names[b] = basis->data.names_data + names_offsets[b];
    i = 0;
    for (o = basis->poly_order; o >= 0 ; o--) {
      d = poly_terms[o] - 1;
      if (d >= 0) {
        basis->data.names[b][i] = 'a'+d;
        i++;
      }
    }
    if (i == 0) {
      basis->data.names[b][0] = '1';
      i++;
    }
    basis->data.names[b][i] = '\0';
    b++;

    /* Add one to the poly_terms data, with carrying. */
    poly_terms[0]++;
    for (o = 0; o < basis->poly_order; o++) {
      if (poly_terms[o] > basis_dim) {
        poly_terms[o+1]++;
        for (PetscInt o2 = o; o2 >= 0; o2--) poly_terms[o2] = poly_terms[o+1];
      } else {
        break;
      }
    }
  }

  for (d = 0; d < basis_dim; d++) {
    for (o = 1; o <= basis->sine_order; o++) {
      basis->data.names[b] = basis->data.names_data + names_offsets[b];
      if (sprintf(basis->data.names[b], "sin(%d*%c)", o, 'a'+d) < 0) {
        PetscFunctionReturn(1);
      }
      b++;
      basis->data.names[b] = basis->data.names_data + names_offsets[b];
      if (sprintf(basis->data.names[b], "cos(%d*%c)", o, 'a'+d) < 0) {
        PetscFunctionReturn(1);
      }
      b++;
    }
  }
  basis->data.max_name_size = 0;
  for (b = 0; b < basis->data.B; b++) {
    basis->data.max_name_size = PetscMax(strlen(basis->data.names[b]), (size_t) basis->data.max_name_size);
  }
  if (basis->poly_order >= 0) {
    ierr = PetscFree(poly_terms);CHKERRQ(ierr);
  }
  ierr = PetscFree(names_offsets);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisCreateData_monolithic(Basis basis) {
  PetscErrorCode   ierr;
  PetscInt         dim;
  const PetscReal  *x;
  Vec              *X;
  PetscInt         n, i, o, d, b;
  PetscReal        *Theta_data;
  PetscInt         *poly_terms;

  PetscFunctionBegin;
  X = basis->data.X;
  dim = basis->data.dim;

  ierr = MatCreateSeqDense(PETSC_COMM_SELF, basis->data.B, basis->data.N, NULL, &basis->data.Theta);CHKERRQ(ierr);
  ierr = MatDenseGetArray(basis->data.Theta, &Theta_data);CHKERRQ(ierr);

  if (basis->poly_order >= 0) {
    ierr = PetscCalloc1(basis->poly_order+1, &poly_terms);CHKERRQ(ierr);
  }

  /* Compute basis data. */
  i = 0;
  for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
    for (o = 0; o <= basis->poly_order; o++) {
      poly_terms[o] = 0;
    }

    /* Iterate through all basis functions of every order up to poly_order. */
    /* Result polynomial is product of entry poly_terms[o]-1 (if it's not 0). */
    ierr = VecGetArrayRead(X[n], &x);CHKERRQ(ierr);
    while (poly_terms[basis->poly_order] < 1) {
      /* Generate the polynomial corresponding to the powers in poly_terms. */
      Theta_data[i] = 1;
      for (o = basis->poly_order; o >= 0 ; o--) {
        d = poly_terms[o] - 1;
        if (d >= 0) {
          Theta_data[i] *= x[d];
        }
      }
      i++;

      /* Add one to the poly_terms data, with carrying. */
      poly_terms[0]++;
      for (o = 0; o < basis->poly_order; o++) {
        if (poly_terms[o] > dim) {
          poly_terms[o+1]++;
          for (PetscInt o2 = o; o2 >= 0; o2--) poly_terms[o2] = poly_terms[o+1];
        } else {
          break;
        }
      }
    }

    for (d = 0; d < dim; d++) {    /* For each degree of freedom d. */
      /* Add trig functions using this degree of freedom. */
      for (o = 1; o <= basis->sine_order; o++) {
        Theta_data[i] = PetscSinReal(o * x[d]);
        i++;
        Theta_data[i] = PetscCosReal(o * x[d]);
        i++;
      }
    }
    ierr = VecRestoreArrayRead(X[n], &x);CHKERRQ(ierr);
  }

  if (basis->normalize_columns) {
    /* Scale the columns to have the same norm. */
    ierr = PetscCalloc1(basis->data.B, &basis->data.column_scales);CHKERRQ(ierr);
    i = 0;
    for (n = 0; n < basis->data.N; n++) {
      for (b = 0; b < basis->data.B; b++) {
        basis->data.column_scales[b] += Theta_data[i]*Theta_data[i];
        i++;
      }
    }
    for (b = 0; b < basis->data.B; b++) {
      basis->data.column_scales[b] = PetscSqrtReal(basis->data.column_scales[b]);
    }
    i = 0;
    for (n = 0; n < basis->data.N; n++) {
      for (b = 0; b < basis->data.B; b++) {
        if (basis->data.column_scales[b]) {
          Theta_data[i] /= basis->data.column_scales[b];
        }
        i++;
      }
    }
  }

  ierr = MatDenseRestoreArray(basis->data.Theta, &Theta_data);CHKERRQ(ierr);
  ierr = MatTranspose(basis->data.Theta,MAT_INPLACE_MATRIX,&(basis->data.Theta));CHKERRQ(ierr);

  ierr = MatAssemblyBegin(basis->data.Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(basis->data.Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (basis->poly_order >= 0) {
    ierr = PetscFree(poly_terms);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisCreateData_individual(Basis basis) {
  PetscErrorCode   ierr;
  PetscInt         dim,B,ctr;
  const PetscReal  *x;
  Vec              *X;
  PetscInt         n, i, o, d, d2, b;
  PetscReal        *Theta_data;
  PetscInt         *poly_terms;
  Mat              Theta;

  PetscFunctionBegin;
  X = basis->data.X;
  dim = basis->data.dim;
  B = basis->data.B;
  ctr = basis->cross_term_range;

  /* Allocate basis data. */
  if (basis->poly_order >= 0) {
    ierr = PetscCalloc1(basis->poly_order+1, &poly_terms);CHKERRQ(ierr);
  }
  if (basis->normalize_columns) {
    ierr = PetscCalloc1(B * dim, &basis->data.column_scales);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(dim, &basis->data.Thetas);CHKERRQ(ierr);

  for (d = 0; d < dim; d++) {    /* For each degree of freedom d. */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, B, basis->data.N, NULL, &Theta);CHKERRQ(ierr);
    basis->data.Thetas[d] = Theta;
    ierr = MatDenseGetArray(Theta, &Theta_data);CHKERRQ(ierr);

    /* Compute basis data. */
    i = 0;
    for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
      for (o = 0; o <= basis->poly_order; o++) {
        poly_terms[o] = 0;
      }

      /* Iterate through all basis functions of every order up to poly_order. */
      /* Result polynomial is product of dimensions poly_terms[o]-1 (if it's not 0). */
      ierr = VecGetArrayRead(X[n], &x);CHKERRQ(ierr);
      while (poly_terms[basis->poly_order] < 1) {
        /* Generate the polynomial corresponding to the powers in poly_terms. */
        Theta_data[i] = 1;
        for (o = basis->poly_order; o >= 0 ; o--) {
          d2 = poly_terms[o] - 1;
          if (d2 >= 0) {
            Theta_data[i] *= x[(d2+d-ctr+dim)%dim];
          }
        }
        i++;

        /* Add one to the poly_terms data, with carrying. */
        poly_terms[0]++;
        for (o = 0; o < basis->poly_order; o++) {
          if (poly_terms[o] > 2*ctr+1) {
            poly_terms[o+1]++;
            for (PetscInt o2 = o; o2 >= 0; o2--) poly_terms[o2] = poly_terms[o+1];
          } else {
            break;
          }
        }
      }

      /* Add trig functions using this degree of freedom and nearby ones. */
      for (d2 = d-ctr; d2 < d+ctr; d2++) {    /* For each nearby degree of freedom d2. */
        for (o = 1; o <= basis->sine_order; o++) {
          Theta_data[i] = PetscSinReal(o * x[(d2+dim)%dim]);
          i++;
          Theta_data[i] = PetscCosReal(o * x[(d2+dim)%dim]);
          i++;
        }
      }
      ierr = VecRestoreArrayRead(X[n], &x);CHKERRQ(ierr);
    }
    if (basis->normalize_columns) {
      /* Scale the columns to have the same norm. */
      i = 0;
      for (n = 0; n < basis->data.N; n++) {
        for (b = 0; b < B; b++) {
          basis->data.column_scales[b+d*B] += Theta_data[i]*Theta_data[i];
          i++;
        }
      }
      for (b = 0; b < B; b++) {
        basis->data.column_scales[b+d*B] = PetscSqrtReal(basis->data.column_scales[b+d*B]);
      }
      i = 0;
      for (n = 0; n < basis->data.N; n++) {
        for (b = 0; b < B; b++) {
          if (basis->data.column_scales[b+d*B]) {
            Theta_data[i] /= basis->data.column_scales[b+d*B];
          }
          i++;
        }
      }
    }
    ierr = MatDenseRestoreArray(Theta, &Theta_data);CHKERRQ(ierr);
    ierr = MatTranspose(Theta,MAT_INPLACE_MATRIX,&Theta);CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Theta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  if (basis->poly_order >= 0) {
    ierr = PetscFree(poly_terms);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



PetscErrorCode SINDyBasisCreateData(Basis basis, Vec* X, PetscInt N)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  /* TODO: either error or free old data before creating new data. */

  /* Get data dimensions. */
  PetscFunctionBegin;
  if (N <= 0) {
    printf("SINDyBasisCreateData(): N should be > 0. Got %d\n", N);
    return 1;
  }

  ierr = VecGetSize(X[0], &dim);CHKERRQ(ierr);
  basis->data.X = X;
  basis->data.N = N;
  basis->data.dim = dim;

  basis->monolithic = basis->cross_term_range == -1;
  if (basis->monolithic) {
    basis->data.B = SINDyCountBases(basis->data.dim, basis->poly_order, basis->sine_order);
    ierr = SINDyBasisCreateData_monolithic(basis);CHKERRQ(ierr);
  } else {
    if (basis->cross_term_range < 0 || 2*basis->cross_term_range+1 > dim) {
      SETERRQ2(PetscObjectComm((PetscObject)X[0]),PETSC_ERR_ARG_WRONG,"Invalid cross_term_range: 2*cross_term_range+1 (=%d) must be between 0 and dim (=%d)", 2*basis->cross_term_range+1, dim);
    }
    basis->data.B = SINDyCountBases(2*basis->cross_term_range+1, basis->poly_order, basis->sine_order);
    ierr = SINDyBasisCreateData_individual(basis);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyFindSparseCoefficients(Basis basis, SparseReg sparse_reg, PetscInt N, Vec* dxdt, PetscInt dim, Vec* Xis)
{
  PetscErrorCode  ierr;
  PetscInt        i,d,k,b;
  PetscInt        old_num_thresholded,num_thresholded;
  PetscBool       *mask;
  PetscReal       *xi_data,*zeros,*dxdt_dim_data;
  Mat             Tcpy;
  PetscInt        *idn, *idb;
  Vec             *dxdt_dim;

  PetscFunctionBegin;
  if (!N) PetscFunctionReturn(0);
  if (N != basis->data.N) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,"the given N (=%d) must match the basis N (=%d)", N, basis->data.N);
  }
  if (dim != basis->data.dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,"the given dim (=%d) must match the basis dim (=%d)", dim, basis->data.dim);
  }

  /* Separate out each dimension of the data. */
  ierr = PetscMalloc1(dim, &dxdt_dim);CHKERRQ(ierr);
  for (d = 0; d < dim; d++) {
    ierr = VecCreateSeq(PETSC_COMM_SELF, N, &dxdt_dim[d]);CHKERRQ(ierr);
    ierr = VecGetArray(dxdt_dim[d], &dxdt_dim_data);CHKERRQ(ierr);
    for (i = 0; i < N; i++) {
      ierr = VecGetValues(dxdt[i], 1, &d, &dxdt_dim_data[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(dxdt_dim[d], &dxdt_dim_data);CHKERRQ(ierr);
  }

  /* Run sparse least squares on each dimension of the data. */
  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_SELF, "SparseReg: dimensions: %d, matrix size: %d x %d\n", dim, basis->data.N, basis->data.B);
  }
  for (d = 0; d < dim; d++) {
    if (sparse_reg->monitor) {
      PetscPrintf(PETSC_COMM_SELF, "SparseReg: dimension: %d, initial least squares\n", d);
    }
    if (basis->monolithic) {
      ierr = SINDySparseLeastSquares(basis->data.Theta, dxdt_dim[d], NULL, Xis[d]);CHKERRQ(ierr);
    } else {
      ierr = SINDySparseLeastSquares(basis->data.Thetas[d], dxdt_dim[d], NULL, Xis[d]);CHKERRQ(ierr);
    }
  }
  if (sparse_reg && sparse_reg->threshold > 0) {
    /* Create a workspace for thresholding. */
    if (basis->monolithic) {
      ierr = MatDuplicate(basis->data.Theta, MAT_COPY_VALUES, &Tcpy);CHKERRQ(ierr);
    } else {
      ierr = MatDuplicate(basis->data.Thetas[0], MAT_COPY_VALUES, &Tcpy);CHKERRQ(ierr);
    }

    ierr = PetscCalloc1(basis->data.N, &zeros);CHKERRQ(ierr);
    ierr = PetscMalloc1(basis->data.B, &mask);CHKERRQ(ierr);
    ierr = PetscMalloc2(basis->data.N, &idn, basis->data.B, &idb);CHKERRQ(ierr);
    for (i=0;i<basis->data.N;i++) idn[i] = i;

    /* Repeatedly threshold and perform least squares on the non-thresholded values. */
    for (d = 0; d < dim; d++) {
      for (i=0;i<basis->data.B;i++) mask[i] = PETSC_FALSE;
      num_thresholded = 0;
      if (basis->monolithic) {
        ierr = MatCopy(basis->data.Theta, Tcpy, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      } else {
        ierr = MatCopy(basis->data.Thetas[d], Tcpy, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }

      for (k = 0; k < sparse_reg->iterations; k++) {
        /* Threshold the data. */
        old_num_thresholded = num_thresholded;
        ierr = VecGetArray(Xis[d], &xi_data);CHKERRQ(ierr);
        for (b = 0; b < basis->data.B; b++) {
          if (!mask[b]) {
            if (PetscAbsReal(xi_data[b]) < sparse_reg->threshold) {
              xi_data[b] = 0;
              idb[num_thresholded] = b;
              num_thresholded++;
              mask[b] = PETSC_TRUE;
            }
          } else {
              xi_data[b] = 0;
          }
        }
        ierr = VecRestoreArray(Xis[d], &xi_data);CHKERRQ(ierr);
        if (sparse_reg->monitor) {
          PetscPrintf(PETSC_COMM_SELF, "SparseReg: dimension: %d, iteration: %d, nonzeros: %d\n", d, k, basis->data.B - num_thresholded);
        }
        if (old_num_thresholded == num_thresholded) break;

        /* Zero out those columns of the matrix. */
        for (b = old_num_thresholded; b < num_thresholded; b++) {
          ierr = MatSetValues(Tcpy,basis->data.N,idn,1,idb+b,zeros,INSERT_VALUES);CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(Tcpy,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Tcpy,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

        /* Run sparse least squares on the non-zero basis functions. */
        /* TODO: I should zero out the right-hand side at the thresholded values, too, so they don't affect the sparsity. */
        ierr = SINDySparseLeastSquares(Tcpy, dxdt_dim[d], NULL, Xis[d]);CHKERRQ(ierr);
      }

      /* Maybe I should zero out the thresholded entries again here, just to make sure Tao didn't mess them up. */
      ierr = VecGetArray(Xis[d], &xi_data);CHKERRQ(ierr);
      for (b = 0; b < num_thresholded; b++) {
        xi_data[idb[b]] = 0;
      }
      ierr = VecRestoreArray(Xis[d], &xi_data);CHKERRQ(ierr);
    }
    ierr = PetscFree(zeros);CHKERRQ(ierr);
    ierr = PetscFree(mask);CHKERRQ(ierr);
    ierr = PetscFree2(idn, idb);CHKERRQ(ierr);
    ierr = MatDestroy(&Tcpy);CHKERRQ(ierr);
  }
  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_WORLD, "SparseReg:%s Xi\n", basis->normalize_columns ? " scaled" : "");
    ierr = SINDyBasisPrint(basis, dim, Xis);
  }
  if (basis->normalize_columns) {
    /* Scale back Xi to the original values. */
    if (basis->monolithic) {
      for (d = 0; d < dim; d++) {
          ierr = VecGetArray(Xis[d], &xi_data);CHKERRQ(ierr);
          for (b = 0; b < basis->data.B; b++) {
            if (basis->data.column_scales[b]) {
              xi_data[b] /= basis->data.column_scales[b];
            }
          }
          ierr = VecRestoreArray(Xis[d], &xi_data);CHKERRQ(ierr);
      }
    } else {
      for (d = 0; d < dim; d++) {
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
      PetscPrintf(PETSC_COMM_WORLD, "SparseReg: Xi\n");
      ierr = SINDyBasisPrint(basis, dim, Xis);
    }
  }

  for (d = 0; d < dim; d++) {
    ierr = VecDestroy(&dxdt_dim[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(dxdt_dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisPrint(Basis basis, PetscInt dim, Vec* Xis)
{
  PetscErrorCode   ierr;
  const PetscReal  **xi_data;
  PetscInt         base_d, max_d, d, b;
  const PetscInt   max_columns = 8;

  PetscFunctionBegin;
  if (dim != basis->data.dim) {
    SETERRQ2(PetscObjectComm((PetscObject)Xis[0]),PETSC_ERR_ARG_WRONG,"the given dim (=%d) must match the basis dim (=%d)", dim, basis->data.dim);
  }

  ierr = SINDyInitializeBasesNames(basis);CHKERRQ(ierr);

  /* Get Xi data. */
  ierr = PetscMalloc1(dim, &xi_data);CHKERRQ(ierr);
  for (d = 0; d < dim; d++) {
    ierr = VecGetArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }

  for (base_d = 0; base_d < dim; base_d += max_columns) {
    max_d = PetscMin(base_d + max_columns, dim);

    /* Print header line. */
    printf("%*s", basis->data.max_name_size+1, "");
    for (d = base_d; d < max_d; d++) {
      printf("   %7s%c/dt", "d", 'a'+d);
    }
    printf("\n");

    /* Print results. */
    for (b = 0; b < basis->data.B; b++) {
      printf("%*s ", basis->data.max_name_size+1, basis->data.names[b]);
      for (d = base_d; d < max_d; d++) {
        if (xi_data[d][b] == 0) {
          printf("   %11s", "0");
        } else {
          printf("   % -9.4e", xi_data[d][b]);
        }
      }
      printf("\n");
    }
    if (max_d != dim) {
      printf("\n");
    }
  }

  /* Restore Xi data. */
  for (d = 0; d < dim; d++) {
    ierr = VecRestoreArrayRead(Xis[d], &xi_data[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree(xi_data);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  Vec            f;               /* solution, function f(x) = A*x-b */
  Mat            J;               /* Jacobian matrix, Transform matrix */
  Tao            tao;                /* Tao solver context */
  PetscReal      hist[100],resid[100];
  PetscInt       lits[100];
  LeastSquaresCtx ctx;
  PetscBool      flg;

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

  ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);

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
  if (var->N) {
    ierr = VecGetSize(vec_data[0], &var->dim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisValidateVariable(Basis basis, Variable var)
{
  PetscFunctionBegin;
  /* Validate variables size. */
  if (var->N != basis->data.N) {
    SETERRQ2(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONGSTATE,
            "Inconsistent data size. Expected var to have size %d but found %d",
            basis->data.N, var->N);
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode SINDyBasisSetOutputVariable(Basis basis, Variable var) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (basis->data.N == -1) basis->data.N = var->N;
  ierr = SINDyBasisValidateVariable(basis, var);CHKERRQ(ierr);
  basis->data.output_var = var;
  PetscFunctionReturn(0);
}

static PetscErrorCode SINDyBasisGetLocalDOF(PetscInt d, PetscInt num_vars, Variable* vars, PetscInt* d_p, Variable* var_p)
{
  PetscErrorCode  ierr;
  PetscInt        v;

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

/* Extracts DOF d2 from the list of variables for data point n. */
static PetscErrorCode SINDyVariableGetDOF(PetscInt d, PetscInt cross_term_range, PetscInt num_vars, Variable* vars,
                                          PetscInt n, PetscInt d2, PetscScalar *val)
{
  PetscErrorCode  ierr;
  Variable        var;
  const PetscReal *x;

  PetscFunctionBegin;
  ierr = SINDyBasisGetLocalDOF(d2, num_vars, vars, &d2, &var);CHKERRQ(ierr);

  /* Need to extract local DOF d2 from variable var. */
  if(var->type == VECTOR) {
    ierr = VecGetArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
    if (cross_term_range != -1 && var->dim <= 2*cross_term_range+1) {
      d2 = (d-cross_term_range + d2 + var->dim) % var->dim;
    }
    *val = x[d2];
    ierr = VecRestoreArrayRead(var->vec_data[n], &x);CHKERRQ(ierr);
  } else if(var->type == SCALAR) {
    *val = var->scalar_data[n];
  } else {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_COR,"Invalid var type %d", var->type);
  }
  PetscFunctionReturn(0);
}

/* Generate name using the given polynomial. */
static PetscErrorCode SINDyBasisGenerateNamePolynomial(Basis basis, PetscInt num_vars, Variable* vars, PetscInt* poly_terms, char** name_p)
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
      if (var->dim == 1) written = sprintf(&name[s], "%s*", var->name);
      else               written = sprintf(&name[s], "%s[%d]*", var->name, d);
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
  } else {
    name_size += 2 + (int) PetscCeilReal(PetscLog10Real(var->dim+2));
    ierr = PetscMalloc1(name_size, &sin_name);CHKERRQ(ierr);
    ierr = PetscMalloc1(name_size, &cos_name);CHKERRQ(ierr);
    if (sprintf(sin_name, "sin(%d*%s[%d])", order, var->name, d) < 0) PetscFunctionReturn(1);
    if (sprintf(cos_name, "cos(%d*%s[%d])", order, var->name, d) < 0) PetscFunctionReturn(1);
  }
  *sin_p = sin_name;
  *cos_p = cos_name;
  PetscFunctionReturn(0);
}

PetscErrorCode SINDyBasisAddVariables(Basis basis, PetscInt num_vars, Variable* vars)
{
  PetscErrorCode  ierr;
  PetscInt        output_dim,input_dim,cross_term_dim,B;
  PetscInt        v, n, i, o, d, d2, b;
  PetscReal       *Theta_data;
  PetscInt        *poly_terms;
  Mat             Theta;
  PetscScalar     val;

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
  output_dim = basis->data.output_var->dim;
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

  for (d = 0; d < output_dim; d++) {
    /* Create a separate matrix for each output degree of freedom d. */
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, basis->data.N, B, NULL, &Theta);CHKERRQ(ierr);
    basis->data.Thetas[d] = Theta;
    ierr = MatDenseGetArray(Theta, &Theta_data);CHKERRQ(ierr);

    b = 0;
    i = 0;
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
      for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
        Theta_data[i] = 1;
        for (o = basis->poly_order; o >= 0 ; o--) {
          d2 = poly_terms[o] - 1;
          if (d2 >= 0) {
            SINDyVariableGetDOF(d, basis->cross_term_range, num_vars, vars, n, d2, &val);
            Theta_data[i] *= val;
          }
        }
        i++;
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
        }
        b += 2;
        for (n = 0; n < basis->data.N; n++) { /* For each data point n. */
          SINDyVariableGetDOF(d, basis->cross_term_range, num_vars, vars, n, d2, &val);
          Theta_data[i] = PetscSinReal(o * val);
          i++;
          Theta_data[i] = PetscCosReal(o * val);
          i++;
        }
      }
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
    for (d = 0; d < var->dim; d++) {
      ierr = VecCreateSeq(PETSC_COMM_SELF, var->N, &dim_vecs[d]);CHKERRQ(ierr);
      ierr = VecGetArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
      for (i = 0; i < var->N; i++) {
        ierr = VecGetValues(var->vec_data[i], 1, &d, &dim_data[i]);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(dim_vecs[d], &dim_data);CHKERRQ(ierr);
    }
  } else if (var->type == SCALAR) {
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, var->N, var->scalar_data, &dim_vecs[0]);CHKERRQ(ierr);
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

  basis->monolithic = PETSC_FALSE;

  /* Run regression on each dimension of the data. */
  if (sparse_reg->monitor) {
    PetscPrintf(PETSC_COMM_SELF, "SINDy: dimensions: %d, matrix size: %d x %d\n", output_dim, basis->data.N, basis->data.B);
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