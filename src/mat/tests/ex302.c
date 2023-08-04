const char help[] = "Test using MatStorageType with MATSEQDENSE";

#include <petscmat.h>
#include <petsc/private/matimpl.h>

// Make a duplicate that uses MAT_STORAGE_FULL
static PetscErrorCode makeFullDuplicate(Mat A, Mat *A_dup)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, A_dup));
  PetscCall(MatSetStorageType(*A_dup, MAT_STORAGE_FULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testStorage(MatStorageType storage, PetscInt m, PetscInt n)
{
  Mat       A, A_full;
  MPI_Comm  comm = PETSC_COMM_SELF;
  PetscBool check;

  PetscFunctionBegin;
  PetscCall(MatCreateDense(comm, m, n, m, n, NULL, &A));
  PetscCall(MatSetStorageType(A, storage));
  PetscCall(PetscObjectSetName((PetscObject)A, "A"));
  PetscCall(MatSetOptionsPrefix(A, "A_"));
  for (PetscInt j = 0; j < n; j++) {
    for (PetscInt i = 0; i < m; i++) {
      PetscReal   diag = i - j;
      PetscScalar v    = 2.0;

      if (diag != 0.0) {
        v = PetscSign(diag) * PetscPowReal(1. / 3., PetscAbsReal(diag));
        if (PetscDefined(USE_COMPLEX)) v *= ((PetscSqrtScalar(-1.0) + 1.0) / PetscSqrtScalar(2.0));
      }
      PetscCall(MatSetValue(A, i, j, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(makeFullDuplicate(A, &A_full));
  PetscCall(PetscObjectSetName((PetscObject)A_full, "A_full"));
  PetscCall(MatSetOptionsPrefix(A_full, "A_full_"));

  // is the full duplicate the same as the original?
  PetscCall(MatEqual(A, A_full, &check));
  PetscCheck(check, comm, PETSC_ERR_PLIB, "Full duplicate of %s matrix not equal to the original", MatStorageTypes[storage]);

  { // convert
    Mat A_sparse, A_full_sparse;

    PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &A_sparse));
    PetscCall(MatConvert(A_full, MATSEQAIJ, MAT_INITIAL_MATRIX, &A_full_sparse));

    PetscCall(MatEqual(A_sparse, A_full_sparse, &check));
    PetscCheck(check, comm, PETSC_ERR_PLIB, "Convert %s", MatStorageTypes[storage]);
    PetscCall(MatDestroy(&A_full_sparse));
    PetscCall(MatDestroy(&A_sparse));
  }

  if (!MatStorageIsUnitDiagonal(storage)) { // axpy test
    Mat          B, B_full;
    PetscInt     ld;
    PetscScalar *b;

    PetscCall(MatDuplicate(A, MAT_SHARE_NONZERO_PATTERN, &B));
    PetscCall(MatDenseGetLDA(B, &ld));
    PetscCall(MatDenseGetArray(B, &b));
    for (PetscInt j = 0; j < n; j++) {
      for (PetscInt i = 0; i < m; i++) {
        b[i + j * ld] = 1.0;
      }
    }
    PetscCall(MatDenseRestoreArray(B, &b));
    PetscCall(makeFullDuplicate(B, &B_full));

    PetscCall(MatAXPY(B, 1.5, A, SAME_NONZERO_PATTERN));
    PetscCall(MatAXPY(B_full, 1.5, A_full, SAME_NONZERO_PATTERN));
    PetscCall(MatEqual(B, B_full, &check));
    PetscCheck(check, comm, PETSC_ERR_PLIB, "AXPY %s", MatStorageTypes[storage]);

    PetscCall(MatDestroy(&B_full));
    PetscCall(MatDestroy(&B));
  }

  if (m == n && MatStorageIsTriangular(storage)) {
    // Compare triangular solve to factorization of full matrix
    Mat A_full_fac;
    Vec b, x, x_full;

    PetscCall(MatCreateVecs(A, &x, &b));
    PetscCall(VecDuplicate(x, &x_full));
    PetscCall(MatGetFactor(A_full, MATSOLVERPETSC, MAT_FACTOR_LU, &A_full_fac));
    PetscCall(MatLUFactorSymbolic(A_full_fac, A_full, NULL, NULL, NULL));
    PetscCall(MatLUFactorNumeric(A_full_fac, A_full, NULL));
    PetscCall(VecSetRandom(b, NULL));
    PetscCall(MatSolve(A, b, x));
    PetscCall(MatSolve(A_full_fac, b, x_full));
    PetscCall(VecEqual(x, x_full, &check));
    PetscCheck(check, comm, PETSC_ERR_PLIB, "AXPY %s", MatStorageTypes[storage]);
    PetscCall(VecDestroy(&x_full));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A_full_fac));
  }

  PetscCall(MatDestroy(&A_full));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MatStorageType storage[] = {
    MAT_STORAGE_FULL,
    MAT_STORAGE_LOWER_TRIANGULAR,
    MAT_STORAGE_UNIT_LOWER_TRIANGULAR,
    MAT_STORAGE_UPPER_TRIANGULAR,
    MAT_STORAGE_UNIT_UPPER_TRIANGULAR,
    MAT_STORAGE_HERMITIAN_LOWER,
    MAT_STORAGE_HERMITIAN_UPPER,
    MAT_STORAGE_SYMMETRIC_LOWER,
    MAT_STORAGE_SYMMETRIC_UPPER,
  };

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  for (PetscInt i = 0; i < PETSC_STATIC_ARRAY_LENGTH(storage); i++) {
    PetscCall(testStorage(storage[i], 4, 4));
    if (!MatStorageIsStructurallySymmetric(storage[i])) {
      PetscCall(testStorage(storage[i], 5, 4));
      PetscCall(testStorage(storage[i], 4, 5));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
