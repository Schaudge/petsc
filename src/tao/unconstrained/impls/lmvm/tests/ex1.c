const char help[] = "Test TAOLMVM on a least-squares problem";

#include <petsctao.h>

typedef struct _n_AppCtx {
  Mat A;
  Vec b;
  Vec r;
} AppCtx;

static PetscErrorCode LSObjAndGrad(Tao tao, Vec x, PetscReal *obj, Vec g, void *_ctx)
{
  PetscFunctionBegin;
  AppCtx *ctx = (AppCtx *) _ctx;
  PetscCall(VecAXPBY(ctx->r, -1.0, 0.0, ctx->b));
  PetscCall(MatMultAdd(ctx->A, x, ctx->r, ctx->r));
  PetscCall(VecDotRealPart(ctx->r, ctx->r, obj));
  *obj *= 0.5;
  PetscCall(MatMultTranspose(ctx->A, ctx->r, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm comm = PETSC_COMM_WORLD;
  AppCtx ctx;

  PetscInt M = 10;
  PetscInt N = 10;
  PetscOptionsBegin(comm, "", help, "TAO");
  PetscCall(PetscOptionsInt("-m", "data size", NULL, M, &M, NULL));
  PetscCall(PetscOptionsInt("-n", "data size", NULL, N, &N, NULL));
  PetscOptionsEnd();

  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &ctx.A));
  Vec sol;

  PetscCall(MatCreateVecs(ctx.A, &sol, &ctx.b));
  PetscCall(VecDuplicate(ctx.b, &ctx.r));
  PetscCall(VecZeroEntries(sol));

  PetscRandom rand;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(ctx.A, rand));
  PetscCall(VecSetRandom(ctx.b, rand));
  PetscCall(PetscRandomDestroy(&rand));

  Tao tao;
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, sol));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, LSObjAndGrad, &ctx));
  PetscCall(TaoSetType(tao, TAOLMVM));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));
  PetscCall(TaoDestroy(&tao));

  PetscCall(VecDestroy(&ctx.r));
  PetscCall(VecDestroy(&sol));
  PetscCall(VecDestroy(&ctx.b));
  PetscCall(MatDestroy(&ctx.A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -tao_monitor -tao_ls_gtol 1.e-6 -tao_view -tao_lmvm_mat_lmvm_hist_size 20

    suffix: 1
    args: -tao_monitor -tao_ls_gtol 1.e-6 -tao_view -tao_lmvm_mat_lmvm_hist_size 20 -tao_lmvm_mat_type lmvmcdbfgs

TEST*/
