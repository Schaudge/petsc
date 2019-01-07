static char help[] = "Simple example to test separable objective optimizers.\n";

#include <petsc.h>
#include <petsctao.h>

typedef struct _UserCtx
{
  PetscInt m;      /* The row dimension of F */
  PetscInt n;      /* The column dimension of F */
  Mat F;           /* matrix in least squares component $(1/2) * || F x - d ||_2^2$ */
  Vec d;           /* RHS in least squares component $(1/2) * || F x - d ||_2^2$ */
  PetscReal alpha; /* regularization constant applied to || x ||_p */
  PetscInt matops;
  NormType p;
} * UserCtx;

PetscErrorCode ConfigureContext(UserCtx ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ctx->m = 10;
  ctx->n = 10;
  ctx->alpha = 1.;
  ctx->matops = 0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "ex4.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m", "The row dimension of matrix F", "ex4.c", ctx->m, &(ctx->m), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n", "The column dimension of matrix F", "ex4.c", ctx->n, &(ctx->n), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matrix_format","Decide format of F matrix. 0 for stencil, 1 for dense random", "ex4.c",
		  					ctx->matops, &(ctx->matops), NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The regularization multiplier", "ex4.c", ctx->alpha, &(ctx->alpha), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char** argv)
{
  UserCtx        ctx;
  PetscErrorCode ierr;
  PetscRandom    rctx;
  PetscInt       Istart,Iend,i,j,Ii,J;
  PetscScalar    v;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ierr = ConfigureContext(ctx);CHKERRQ(ierr);

  /* Creating random ctx */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx); CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);

  /* build the matrix F in ctx */
  ierr = MatCreate(PETSC_COMM_WORLD, &(ctx->F)); CHKERRQ(ierr);
  ierr = MatSetSizes(ctx->F,PETSC_DECIDE, PETSC_DECIDE, ctx->m, ctx->n);CHKERRQ(ierr);
  ierr = MatSetType(ctx->F,MATDUMMY); CHKERRQ(ierr); /* TODO: Decide specific SetType other than dummy*/
  ierr = MatMPIAIJSetPreallocation(ctx->F, 5, NULL, 5, NULL); CHKERRQ(ierr); /*TODO: some number other than 5?*/
  ierr = MatSeqAIJSetPreallocation(ctx->F, 5, NULL); CHKERRQ(ierr);
  ierr = MatSetUp(ctx->F); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(ctx->F,&Istart,&Iend); CHKERRQ(ierr);  


  ierr = PetscLogStageRegister("Assembly", &stage); CHKERRQ(ierr);
  ierr= PetscLogStagePush(stage); CHKERRQ(ierr);

  /* Set matrix elements in  2-D five-point stencil format. See ksp ex02 */
  
  if (!(ctx->matops)){
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; i = Ii/ctx->n; j = Ii - i*ctx->n;
      if (i>0)   {J = Ii - ctx->n; MatSetValues(ctx->F,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (i<ctx->m-1) {J = Ii + ctx->n; MatSetValues(ctx->F,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (j>0)   {J = Ii - 1; MatSetValues(ctx->F,1,&Ii,1,&J,&v,ADD_VALUES);}
      if (j<ctx->n-1) {J = Ii + 1; MatSetValues(ctx->F,1,&Ii,1,&J,&v,ADD_VALUES);}
      v = 4.0; MatSetValues(ctx->F,1,&Ii,1,&Ii,&v,ADD_VALUES);
    }
  }
  else {
    ierr = MatSetRandom(ctx->F, rctx); CHKERRQ(ierr);

  }
  ierr = MatAssemblyBegin(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->F, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscLogStagePop(); CHKERRQ(ierr);

  /* Stencil matrix is symmetric. Setting symmetric flag for ICC/CHolesky preconditioner */
  if (!(ctx->matops)){
    ierr = MatSetOption(ctx->F,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
  }

  /* build the rhs d in ctx */
  /* Define two functions that could pass as objectives to TaoSetObjectiveRoutine(): one
   * for the misfit component, and one for the regularization component */

  ierr = VecCreate(PETSC_COMM_WORLD,&(ctx->d)); CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->d,PETSC_DECIDE,ctx->m); CHKERRQ(ierr);
  ierr=  VecSetFromOptions(ctx->d); CHKERRQ(ierr);
  
  ierr = VecSetRandom(ctx->d,rctx); CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);

  /* Define a single function that calls both components adds them together: the complete objective,
   * in the absence of a Tao implementation that handles separability */

  /* Construct the Tao object */

  /* solve */

  /* examine solution */
  ierr = MatDestroy(&(ctx->F)); CHKERRQ(ierr);
  ierr = VecDestroy(&(ctx->d)); CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args:

TEST*/
