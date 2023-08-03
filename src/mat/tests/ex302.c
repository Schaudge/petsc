const char help[] = "Test using MatStorageType with MATSEQDENSE";

#include <petscmat.h>

static PetscErrorCode makeDuplicate(Mat A, Mat *A_dup)
{
  PetscInt m, n;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(MatCreateDense(PETSC_COMM_SELF, m, n, m, n, NULL, A_dup));
  for (PetscInt j = 0; j < n; j++) {
    for (PetscInt i = 0; i < m; i++) {
      PetscScalar v;

      PetscCall(MatGetValue(A, i, j, &v));
      PetscCall(MatSetValue(*A_dup, i, j, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(*A_dup, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A_dup, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testStorage(MatStorageType storage, PetscInt m, PetscInt n)
{
  Mat A, A_full;

  PetscFunctionBegin;
  PetscCall(MatCreateDense(PETSC_COMM_SELF, m, n, m, n, NULL, &A));
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
  PetscCall(makeDuplicate(A, &A_full));
  PetscCall(PetscObjectSetName((PetscObject)A_full, "A_full"));
  PetscCall(MatSetOptionsPrefix(A_full, "A_full_"));
  PetscCall(MatDestroy(&A_full));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  MatStorageType storage[] = {
    MAT_STORAGE_ALL,
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
    PetscCall(testStorage(storage[i], 5, 4));
    PetscCall(testStorage(storage[i], 4, 5));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
