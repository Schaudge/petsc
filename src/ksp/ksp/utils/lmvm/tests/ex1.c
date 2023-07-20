const char help[] = "Test correctness of Broyden methods";

#include <petscksp.h>

static PetscErrorCode MatSolveHermitianTranspose(Mat B, Vec x, Vec y)
{
  PetscFunctionBegin;
  Vec x_conj;
  PetscCall(VecDuplicate(x, &x_conj));
  PetscCall(VecCopy(x, x_conj));
  PetscCall(VecConjugate(x_conj));
  PetscCall(MatSolveTranspose(B, x_conj, y));
  PetscCall(VecDestroy(&x_conj));
  PetscCall(VecConjugate(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HermitianTransposeTest(Mat B, PetscRandom rand, PetscBool inverse)
{
  PetscFunctionBegin;
  Vec x, f, Bx, Bhf;
  PetscCall(MatCreateVecs(B, &x, &f));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(f, rand));
  PetscCall(MatCreateVecs(B, &Bhf, &Bx));
  PetscCall((inverse ? MatSolve : MatMult)(B, x, Bx));
  PetscCall((inverse ? MatSolveHermitianTranspose : MatMultHermitianTranspose)(B, f, Bhf));
  PetscScalar dot_a, dot_b;
  PetscReal   x_norm, Bhf_norm, Bx_norm, f_norm;
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(Bhf, NORM_2, &Bhf_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecNorm(f, NORM_2, &f_norm));
  PetscCall(VecDot(x, Bhf, &dot_a));
  PetscCall(VecDot(Bx, f, &dot_b));
  PetscReal err   = PetscAbsScalar(dot_a - dot_b);
  PetscReal scale = PetscMax(x_norm * Bhf_norm, Bx_norm * f_norm);
  PetscCall(PetscInfo((PetscObject)B, "Hermitian transpose error %g, relative error %g \n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Hermitian transpose error %g", (double)err);
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&Bhf));
  PetscFunctionReturn(0);
}

static PetscErrorCode InverseTest(Mat B, PetscRandom rand)
{
  PetscFunctionBegin;
  Vec x, Bx, BinvBx;
  PetscCall(MatCreateVecs(B, &x, &Bx));
  PetscCall(VecDuplicate(x, &BinvBx));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(MatMult(B, x, Bx));
  PetscCall(MatSolve(B, Bx, BinvBx));
  PetscReal x_norm, Bx_norm, err;
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecAXPY(BinvBx, -1.0, x));
  PetscCall(VecNorm(BinvBx, NORM_2, &err));
  PetscReal scale = PetscMax(x_norm, Bx_norm);
  PetscCall(PetscInfo((PetscObject)B, "Inverse error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > 10.0 * PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Inverse error %g", (double)err);
  PetscCall(VecDestroy(&BinvBx));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsHermitianTest(Mat B, PetscRandom rand)
{
  PetscFunctionBegin;
  Vec x, y, Bx, By;
  PetscCall(MatCreateVecs(B, &x, &y));
  PetscCall(VecDuplicate(x, &By));
  PetscCall(VecDuplicate(y, &Bx));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(y, rand));
  PetscCall(MatMult(B, x, Bx));
  PetscCall(MatMult(B, y, By));
  PetscScalar dot_a, dot_b;
  PetscReal   x_norm, By_norm, Bx_norm, y_norm;
  PetscCall(VecNorm(x, NORM_2, &x_norm));
  PetscCall(VecNorm(By, NORM_2, &By_norm));
  PetscCall(VecNorm(Bx, NORM_2, &Bx_norm));
  PetscCall(VecNorm(y, NORM_2, &y_norm));
  PetscCall(VecDot(x, By, &dot_a));
  PetscCall(VecDot(Bx, y, &dot_b));
  PetscReal err   = PetscAbsScalar(dot_a - dot_b);
  PetscReal scale = PetscMax(x_norm * By_norm, Bx_norm * y_norm);
  PetscCall(PetscInfo((PetscObject)B, "Is Hermitian error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Is Hermitian error %g", (double)err);
  PetscCall(VecDestroy(&By));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SecantTest(Mat B, Vec dx, Vec df, PetscBool is_hermitian)
{
  PetscFunctionBegin;
  Vec B_x;
  PetscCall(VecDuplicate(df, &B_x));
  PetscCall(MatMult(B, dx, B_x));
  PetscCall(VecAXPY(B_x, -1.0, df));
  PetscReal err, scale;
  PetscCall(VecNorm(B_x, NORM_2, &err));
  PetscCall(VecNorm(df, NORM_2, &scale));
  PetscCall(PetscInfo((PetscObject)B, "Secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Secant error %g", (double)err);

  if (is_hermitian) {
    PetscCall(MatMultHermitianTranspose(B, dx, B_x));
    PetscCall(VecAXPY(B_x, -1.0, df));
    PetscReal err;
    PetscCall(VecNorm(B_x, NORM_2, &err));
    PetscCall(PetscInfo((PetscObject)B, "Adjoint secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
    if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Adjoint secant error %g", (double)err);
  }

  PetscLayout rmap, cmap;
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscBool square;
  PetscCall(PetscLayoutCompare(rmap, cmap, &square));
  if (square) {
    PetscCall(MatSolve(B, df, B_x));
    PetscCall(VecAXPY(B_x, -1.0, dx));

    PetscReal err;
    PetscCall(VecNorm(B_x, NORM_2, &err));
    PetscCall(VecNorm(dx, NORM_2, &scale));
    PetscCall(PetscInfo((PetscObject)B, "Inverse secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
    if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Inverse secant error %g", (double)err);

    if (is_hermitian) {
      PetscCall(MatSolveHermitianTranspose(B, df, B_x));
      PetscCall(VecAXPY(B_x, -1.0, dx));
      PetscReal err;
      PetscCall(VecNorm(B_x, NORM_2, &err));
      PetscCall(PetscInfo((PetscObject)B, "Adjoint inverse secant error %g, relative error %g\n", (double)err, (double)(err / scale)));
      if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_PLIB, "Adjoint inverse secant error %g", (double)err);
    }
  }
  PetscCall(VecDestroy(&B_x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  TEST_BRDN,
  TEST_SYMBRDN,
  TEST_SYMBRDNPHI,
  TEST_SR1
} TestType;

static PetscErrorCode RankOneAXPY(Mat C, PetscScalar alpha, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscInt m, n, M, N;

  PetscCall(MatGetSize(C, &M, &N));
  PetscCall(MatGetLocalSize(C, &m, &n));

  Mat col_mat, row_mat;
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)C), m, PETSC_DECIDE, M, 1, NULL, &col_mat));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)C), n, PETSC_DECIDE, N, 1, NULL, &row_mat));

  const PetscScalar *x_a, *y_a;
  PetscCall(VecGetArrayRead(x, &x_a));
  PetscCall(VecGetArrayRead(y, &y_a));

  PetscScalar *x_mat_a, *y_mat_a;
  PetscCall(MatDenseGetColumn(col_mat, 0, &x_mat_a));
  PetscCall(MatDenseGetColumn(row_mat, 0, &y_mat_a));

  PetscCall(PetscArraycpy(x_mat_a, x_a, m));
  PetscCall(PetscArraycpy(y_mat_a, y_a, n));

  PetscCall(MatDenseRestoreColumn(row_mat, &y_mat_a));
  PetscCall(MatDenseRestoreColumn(col_mat, &x_mat_a));

  PetscCall(VecRestoreArrayRead(y, &y_a));
  PetscCall(VecRestoreArrayRead(x, &x_a));

  Mat outer_product;
  PetscCall(MatConjugate(row_mat));
  PetscCall(MatMatTransposeMult(col_mat, row_mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &outer_product));

  PetscCall(MatAXPY(C, alpha, outer_product, SAME_NONZERO_PATTERN));

  PetscCall(MatDestroy(&outer_product));
  PetscCall(MatDestroy(&row_mat));
  PetscCall(MatDestroy(&col_mat));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BroydenUpdateTest(Mat B_k, Mat B_kplus, Vec dx, Vec df)
{
  PetscFunctionBegin;
  PetscInt m, n, M, N;

  PetscCall(MatGetSize(B_k, &M, &N));
  PetscCall(MatGetLocalSize(B_k, &m, &n));
  Mat D;
  Vec v;
  PetscCall(VecDuplicate(df, &v));
  PetscCall(MatDuplicate(B_kplus, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B_k, SAME_NONZERO_PATTERN));
  PetscCall(MatMult(D, dx, v));
  PetscReal dx_norm_2;
  PetscCall(VecDotRealPart(dx, dx, &dx_norm_2));
  PetscCall(VecScale(v, 1.0 / dx_norm_2));

  PetscReal scale;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &scale));
  scale += 1;

  PetscCall(RankOneAXPY(D, -1.0, v, dx));

  PetscReal err;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &err));
  PetscCall(PetscInfo((PetscObject)B_k, "Broyden update error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL) SETERRQ(PetscObjectComm((PetscObject)B_k), PETSC_ERR_PLIB, "Broyden update error %g", (double)err);

  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymmetricBroydenUpdateTest(Mat B_k, Mat B_kplus, Vec dx, Vec df)
{
  PetscFunctionBegin;
  PetscInt n, N;

  PetscCall(MatGetSize(B_k, NULL, &N));
  PetscCall(MatGetLocalSize(B_k, NULL, &n));
  Mat         D;
  Vec         y;
  Vec         s;
  Vec         p;
  PetscScalar rho;
  PetscCall(VecDot(dx, df, &rho));
  PetscScalar sqrt_rho = 1.0 / PetscSqrtScalar(rho);
  PetscCall(VecDuplicate(df, &y));
  PetscCall(VecCopy(df, y));
  PetscCall(VecDuplicate(dx, &s));
  PetscCall(VecCopy(dx, s));
  PetscCall(VecScale(y, sqrt_rho));
  PetscCall(VecScale(s, sqrt_rho));
  PetscCall(VecDuplicate(y, &p));
  PetscCall(MatMult(B_k, s, p));
  PetscScalar stp;
  PetscCall(VecDot(p, s, &stp));
  PetscCall(MatDuplicate(B_kplus, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B_k, SAME_NONZERO_PATTERN));

  PetscReal scale;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &scale));
  scale += 1;

  PetscCall(RankOneAXPY(D, -(1.0 + stp), y, y));
  PetscCall(RankOneAXPY(D, 1.0, p, y));
  PetscCall(RankOneAXPY(D, 1.0, y, p));

  PetscReal err;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &err));
  PetscCall(PetscInfo((PetscObject)B_k, "Symmetric Broyden update error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B_k), PETSC_ERR_PLIB, "Symmetric Broyden update error %g", (double)err);

  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SymmetricBroydenPhiUpdateTest(Mat B_k, Mat B_kplus, PetscReal phi, Vec dx, Vec df)
{
  PetscFunctionBegin;
  PetscInt n, N;

  PetscCall(MatGetSize(B_k, NULL, &N));
  PetscCall(MatGetLocalSize(B_k, NULL, &n));
  Mat         D;
  Vec         y;
  Vec         s;
  Vec         v;
  Vec         p;
  PetscScalar rho;
  PetscCall(VecDot(dx, df, &rho));
  PetscCall(VecDuplicate(df, &y));
  PetscCall(VecCopy(df, y));
  PetscCall(VecDuplicate(dx, &s));
  PetscCall(VecCopy(dx, s));
  PetscCall(VecDuplicate(y, &p));
  PetscCall(MatMult(B_k, s, p));
  PetscScalar stp;
  PetscCall(VecDot(p, s, &stp));
  PetscCall(VecDuplicate(y, &v));
  PetscCall(VecCopy(y, v));
  PetscCall(VecAXPBY(v, -1.0 / stp, 1.0 / rho, p));
  PetscCall(MatDuplicate(B_kplus, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B_k, SAME_NONZERO_PATTERN));

  PetscReal scale;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &scale));
  scale += 1;

  PetscCall(RankOneAXPY(D, 1.0 / stp, p, p));
  PetscCall(RankOneAXPY(D, -1.0 / rho, y, y));
  PetscCall(RankOneAXPY(D, -phi * stp, v, v));

  PetscReal err;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &err));
  PetscCall(PetscInfo((PetscObject)B_k, "Symmetric Broyden phi update error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B_k), PETSC_ERR_PLIB, "Symmetric Broyden phi update error %g", (double)err);

  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SR1UpdateTest(Mat B_k, Mat B_kplus, Vec dx, Vec df)
{
  PetscFunctionBegin;
  PetscInt n, N;

  PetscCall(MatGetSize(B_k, NULL, &N));
  PetscCall(MatGetLocalSize(B_k, NULL, &n));
  Mat D;
  Vec p;
  PetscCall(VecDuplicate(df, &p));
  PetscCall(MatMult(B_k, dx, p));
  PetscCall(VecAYPX(p, -1.0, df));
  PetscScalar stp;
  PetscCall(VecDot(p, dx, &stp));
  PetscCall(MatDuplicate(B_kplus, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B_k, SAME_NONZERO_PATTERN));

  PetscReal scale;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &scale));
  scale += 1;

  PetscCall(RankOneAXPY(D, -1.0 / stp, p, p));

  PetscReal err;
  PetscCall(MatNorm(D, NORM_FROBENIUS, &err));
  PetscCall(PetscInfo((PetscObject)B_k, "SR1 update error %g, relative error %g\n", (double)err, (double)(err / scale)));
  if (err > PETSC_SMALL * scale) SETERRQ(PetscObjectComm((PetscObject)B_k), PETSC_ERR_PLIB, "SR1 update error %g", (double)err);

  PetscCall(MatDestroy(&D));
  PetscCall(VecDestroy(&p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Solve(Mat A, Vec x, Vec y)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, (void *)&B));
  PetscCall(MatSolve(B, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatComputeInverseOperator(Mat B, Mat *B_k)
{
  PetscFunctionBegin;
  Mat      Binv;
  PetscInt m, n, M, N;

  PetscCall(MatGetSize(B, &M, &N));
  PetscCall(MatGetLocalSize(B, &m, &n));
  PetscCall(MatCreateShell(PetscObjectComm((PetscObject)B), m, n, M, N, (void *)B, &Binv));
  PetscCall(MatShellSetOperation(Binv, MATOP_MULT, (void (*)(void))MatMult_Solve));
  PetscCall(MatComputeOperator(Binv, MATDENSE, B_k));
  PetscCall(MatDestroy(&Binv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestUpdate(Mat B, PetscRandom rand, PetscBool is_hermitian, PetscBool inverse, TestType test_type, PetscBool nilpotent_iter)
{
  PetscFunctionBegin;
  PetscLayout rmap, cmap;
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscBool is_square;
  PetscCall(PetscLayoutCompare(rmap, cmap, &is_square));

  Mat B_k = NULL;
  if (inverse) {
    if (is_square) PetscCall(MatComputeInverseOperator(B, &B_k));
  } else {
    PetscCall(MatComputeOperator(B, MATDENSE, &B_k));
  }

  Vec x, dx, f, x_prev, f_prev, df;
  PetscCall(MatLMVMGetLastUpdate(B, &x_prev, &f_prev));
  PetscCall(VecDuplicate(x_prev, &x));
  PetscCall(VecDuplicate(x_prev, &dx));
  PetscCall(VecDuplicate(f_prev, &df));
  PetscCall(VecDuplicate(f_prev, &f));
  PetscCall(VecSetRandom(dx, rand));
  if (nilpotent_iter) {
    Mat J0;

    PetscCall(MatLMVMGetJ0(B, &J0));
    PetscCall(MatMult(J0, dx, df));
  } else {
    PetscCall(VecSetRandom(df, rand));

    PetscScalar rho;
    if (is_square) {
      PetscCall(VecDot(dx, df, &rho));
      PetscCall(VecScale(dx, PetscAbsScalar(rho) / rho));
    }
  }
  PetscCall(VecWAXPY(x, 1.0, x_prev, dx));
  PetscCall(VecWAXPY(f, 1.0, f_prev, df));
  PetscCall(MatLMVMUpdate(B, x, f));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));

  Mat B_kplus = NULL;
  if (inverse) {
    if (is_square) PetscCall(MatComputeInverseOperator(B, &B_kplus));
  } else {
    PetscCall(MatComputeOperator(B, MATDENSE, &B_kplus));
  }

  PetscCall(SecantTest(B, dx, df, is_hermitian));
  PetscCall(HermitianTransposeTest(B, rand, PETSC_FALSE));
  if (is_hermitian) PetscCall(IsHermitianTest(B, rand));

  if (is_square) {
    PetscCall(InverseTest(B, rand));
    PetscCall(HermitianTransposeTest(B, rand, PETSC_TRUE));
  }

  switch (test_type) {
  case TEST_BRDN:
    if (inverse) {
      if (is_square) PetscCall(BroydenUpdateTest(B_k, B_kplus, df, dx));
    } else {
      PetscCall(BroydenUpdateTest(B_k, B_kplus, dx, df));
    }
    break;
  case TEST_SYMBRDN:
    if (inverse) PetscCall(SymmetricBroydenUpdateTest(B_k, B_kplus, df, dx));
    else PetscCall(SymmetricBroydenUpdateTest(B_k, B_kplus, dx, df));
    break;
  case TEST_SYMBRDNPHI: {
    PetscReal phi;

    if (inverse) {
      PetscCall(MatLMVMSymBadBroydenGetPsi(B, &phi));
    } else {
      PetscCall(MatLMVMSymBroydenGetPhi(B, &phi));
    }
    PetscCheck(phi >= 0.0 && phi <= 1.0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "phi %g is not in [0,1]", (double)phi);
    if (inverse) PetscCall(SymmetricBroydenPhiUpdateTest(B_k, B_kplus, phi, df, dx));
    else PetscCall(SymmetricBroydenPhiUpdateTest(B_k, B_kplus, phi, dx, df));
  } break;
  case TEST_SR1:
    PetscCall(SR1UpdateTest(B_k, B_kplus, dx, df));
    break;
  }

  PetscCall(VecDestroy(&dx));
  PetscCall(VecDestroy(&df));
  PetscCall(MatDestroy(&B_k));
  PetscCall(MatDestroy(&B_kplus));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscInt M = 15, N = 15, hist_size = 5, n_iter = 3 * hist_size;
  PetscInt n_nilpotent_iter = 0;
  MPI_Comm comm             = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm, NULL, help, NULL);
  PetscCall(PetscOptionsInt("-m", "# Matrix rows", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "# Matrix columns", NULL, N, &N, NULL));
  PetscCall(PetscOptionsInt("-n_iter", "# test iterations", NULL, n_iter, &n_iter, NULL));
  PetscCall(PetscOptionsInt("-n_nilpotent_iter", "# test iterations where the update doesn't change the matrix", NULL, n_nilpotent_iter, &n_nilpotent_iter, NULL));
  PetscOptionsEnd();

  Mat B;
  PetscCall(MatCreate(comm, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, M, N));
  PetscCall(MatSetOptionsPrefix(B, "B_"));
  PetscCall(KSPInitializePackage());
  PetscCall(MatSetType(B, MATLMVMBROYDEN));
  PetscCall(MatLMVMSetHistorySize(B, hist_size));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));

  PetscBool B_is_h, B_is_h_known;
  PetscCall(MatIsHermitianKnown(B, &B_is_h_known, &B_is_h));
  PetscBool is_hermitian = (B_is_h_known && B_is_h) ? PETSC_TRUE : PETSC_FALSE;

  Mat J0;
  PetscCall(MatLMVMGetJ0(B, &J0));
  PetscCall(MatSetType(J0, MATCONSTANTDIAGONAL));
  PetscCall(MatSetFromOptions(J0));
  PetscCall(MatSetUp(J0));

  PetscBool is_constantdiag, is_vectordiag;
  PetscCall(PetscObjectTypeCompare((PetscObject)J0, MATCONSTANTDIAGONAL, &is_constantdiag));
  PetscCall(PetscObjectTypeCompare((PetscObject)J0, MATDIAGONAL, &is_vectordiag));

  PetscRandom rand;
  PetscCall(PetscRandomCreate(comm, &rand));
  if (PetscDefined(USE_COMPLEX)) PetscCall(PetscRandomSetInterval(rand, -1.0 - PetscSqrtScalar(-1.0), 1.0 + PetscSqrtScalar(-1.0)));
  else PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  if (is_constantdiag) {
    PetscScalar diag;
    PetscCall(PetscRandomGetValue(rand, &diag));
    PetscCallMPI(MPI_Bcast(&diag, 1, MPIU_SCALAR, 0, comm));
    PetscCall(MatLMVMSetJ0Scale(B, 1.0));
  } else if (is_vectordiag) {
    Vec diag;
    PetscCall(MatCreateVecs(B, &diag, NULL));
    PetscCall(VecSetRandom(diag, NULL));

    if (is_hermitian) {
      Vec diag_conj;

      PetscCall(VecDuplicate(diag, &diag_conj));
      PetscCall(VecCopy(diag, diag_conj));
      PetscCall(VecConjugate(diag_conj));
      PetscCall(VecPointwiseMult(diag, diag, diag_conj));
      PetscCall(VecDestroy(&diag_conj));
    }

    PetscCall(MatLMVMSetJ0Diag(B, diag));
    PetscCall(VecDestroy(&diag));
  } else {
    PetscCall(MatSetRandom(J0, rand));

    if (is_hermitian) {
      Mat J0H;

      PetscCall(MatHermitianTranspose(J0, MAT_INITIAL_MATRIX, &J0H));
      PetscCall(MatAXPY(J0, 1.0, J0H, SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&J0H));

      Mat J0copy;

      PetscCall(MatDuplicate(J0, MAT_COPY_VALUES, &J0copy));

      PetscReal *real_eig, *imag_eig;
      PetscCall(PetscMalloc2(N, &real_eig, N, &imag_eig));
      KSP kspeig;
      PetscCall(KSPCreate(comm, &kspeig));
      PetscCall(KSPSetType(kspeig, KSPCG));
      PetscCall(KSPSetPCSide(kspeig, PC_SYMMETRIC));
      PC pceig;
      PetscCall(KSPGetPC(kspeig, &pceig));
      PetscCall(PCSetType(pceig, PCNONE));
      PetscCall(KSPSetOperators(kspeig, J0copy, J0copy));
      PetscCall(KSPComputeEigenvaluesExplicitly(kspeig, N, real_eig, imag_eig));
      PetscCallMPI(MPI_Bcast(real_eig, N, MPIU_REAL, 0, comm));
      PetscCall(PetscSortReal(N, real_eig));
      PetscReal shift = PetscMax(2 * PetscAbsReal(real_eig[N - 1]), 2 * PetscAbsReal(real_eig[0]));
      PetscCall(MatShift(J0, shift));
      PetscCall(MatAssemblyBegin(J0, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(J0, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscFree2(real_eig, imag_eig));
      PetscCall(KSPDestroy(&kspeig));
      PetscCall(MatDestroy(&J0copy));
    }
    PetscCall(MatLMVMSetJ0(B, J0));
  }

  PetscCall(MatViewFromOptions(B, NULL, "-view"));
  PetscCall(MatViewFromOptions(J0, NULL, "-view"));

  TestType test_type = TEST_BRDN;

  PetscBool is_symbrdn;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &is_symbrdn, MATLMVMDFP, MATLMVMBFGS, ""));
  if (is_symbrdn) { test_type = TEST_SYMBRDN; }

  PetscBool is_sr1;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMSR1, &is_sr1));
  if (is_sr1) { test_type = TEST_SR1; }

  PetscBool is_symbrdnphi;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &is_symbrdnphi, MATLMVMSYMBROYDEN, MATLMVMSYMBADBROYDEN, ""));
  if (is_symbrdnphi) { test_type = TEST_SYMBRDNPHI; }

  PetscBool inverse = PETSC_FALSE;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &inverse, MATLMVMBADBROYDEN, MATLMVMBFGS, MATLMVMSYMBADBROYDEN, ""));

  // Initialize with the first location
  Vec x, f;
  PetscCall(MatCreateVecs(B, &x, &f));
  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(f, rand));
  PetscCall(MatLMVMUpdate(B, x, f));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));

  PetscCall(HermitianTransposeTest(B, rand, PETSC_FALSE));
  if (is_hermitian) PetscCall(IsHermitianTest(B, rand));

  PetscLayout rmap, cmap;
  PetscCall(MatGetLayouts(B, &rmap, &cmap));
  PetscBool is_square;
  PetscCall(PetscLayoutCompare(rmap, cmap, &is_square));
  if (is_square) {
    PetscCall(InverseTest(B, rand));
    PetscCall(HermitianTransposeTest(B, rand, PETSC_TRUE));
  }

  for (PetscInt i = 0; i < n_iter; i++) PetscCall(TestUpdate(B, rand, is_hermitian, inverse, test_type, (i < n_nilpotent_iter) ? PETSC_TRUE : PETSC_FALSE));

  PetscCall(PetscRandomDestroy(&rand));

  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: broyden_rectangular
    nsize: 2
    args: -m 15 -n 10 -n_iter 8 -n_nilpotent_iter 3 -B_lmvm_J0_mat_type dense -B_mat_lmvm_matvec_type {{recursive compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}

  test:
    suffix: square
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type {{lmvmbroyden lmvmbadbroyden}} -B_mat_lmvm_matvec_type {{recursive compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}
    args: -B_lmvm_J0_mat_type dense -B_lmvm_J0_pc_type bjacobi -B_lmvm_J0_sub_pc_type lu -B_lmvm_J0_ksp_type gmres -B_lmvm_J0_ksp_max_it 15 -B_lmvm_J0_ksp_rtol 0.0 -B_lmvm_J0_ksp_atol 0.0

  test:
    suffix: square_sr1
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 5 -B_mat_type lmvmsr1 -B_mat_lmvm_matvec_type {{recursive compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}
    args: -B_lmvm_J0_mat_type dense -B_lmvm_J0_pc_type bjacobi -B_lmvm_J0_sub_pc_type lu -B_lmvm_J0_ksp_type gmres -B_lmvm_J0_ksp_max_it 15 -B_lmvm_J0_ksp_rtol 0.0 -B_lmvm_J0_ksp_atol 0.0

  test:
    suffix: square_symmetric
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type {{lmvmdfp lmvmbfgs}} -B_lmvm_J0_mat_type dense -B_lmvm_J0_pc_type bjacobi -B_lmvm_J0_sub_pc_type lu -B_lmvm_J0_ksp_type gmres -B_mat_lmvm_scale_type user -B_lmvm_J0_ksp_max_it 15 -B_lmvm_J0_ksp_rtol 0.0 -B_lmvm_J0_ksp_atol 0.0

  test:
    suffix: square_symmetric_phi
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type lmvmsymbroyden -B_mat_lmvm_phi {{0.0 0.6 1.0}} -B_lmvm_J0_mat_type dense -B_lmvm_J0_pc_type bjacobi -B_lmvm_J0_sub_pc_type lu -B_lmvm_J0_ksp_type gmres -B_mat_lmvm_scale_type user -B_lmvm_J0_ksp_max_it 15 -B_lmvm_J0_ksp_rtol 0.0 -B_lmvm_J0_ksp_atol 0.0

  test:
    suffix: square_symmetric_psi
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type lmvmsymbadbroyden -B_mat_lmvm_psi {{0.0 0.6 1.0}} -B_lmvm_J0_mat_type dense -B_lmvm_J0_pc_type bjacobi -B_lmvm_J0_sub_pc_type lu -B_lmvm_J0_ksp_type gmres -B_mat_lmvm_scale_type user -B_lmvm_J0_ksp_max_it 15 -B_lmvm_J0_ksp_rtol 0.0 -B_lmvm_J0_ksp_atol 0.0

  test:
    suffix: square_diag
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type {{lmvmbroyden lmvmbadbroyden}} -B_lmvm_J0_mat_type {{constantdiagonal diagonal}}
    args: -B_mat_lmvm_matvec_type {{recursive compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}

  test:
    suffix: square_diag_sr1
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 5 -B_mat_type lmvmsr1 -B_lmvm_J0_mat_type {{constantdiagonal diagonal}}
    args: -B_mat_lmvm_matvec_type {{recursive compact_dense}} -B_mat_lmvm_cache_J0_products {{false true}}

  test:
    suffix: square_diag_symmetric
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type {{lmvmdfp lmvmbfgs}} -B_lmvm_J0_mat_type {{constantdiagonal diagonal}} -B_mat_lmvm_scale_type user

  test:
    suffix: square_symmetric_phi_diag
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type lmvmsymbroyden -B_mat_lmvm_phi {{0.0 0.6 1.0}} -B_lmvm_J0_mat_type {{constantdiagonal diagonal}} -B_mat_lmvm_scale_type user

  test:
    suffix: square_symmetric_psi_diag
    output_file: output/ex1_square.out
    nsize: 2
    args: -m 15 -n 15 -n_iter 8 -n_nilpotent_iter 3 -B_mat_type lmvmsymbadbroyden -B_mat_lmvm_psi {{0.0 0.6 1.0}} -B_lmvm_J0_mat_type {{constantdiagonal diagonal}} -B_mat_lmvm_scale_type user

TEST*/
