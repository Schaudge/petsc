#include <petsctao.h>
#include <petscmat.h>
#include <petscksp.h>
/*
Description:   VMNOS tomography reconstruction example .
               0.5*||Ax-b||^2 + lambda*g(x) s.t. x >=0.
               Turning constraint to barrier term,
               0.5*||Ax-b||^2 + lambda*g(x) - mu*sum(log(x_i))
Reference:     ADMM and BRGN Tomography Example
*/

static char help[] = "Finds the VMNOS solution to the under constraint linear model Ax = b s.t. x >= 0, with regularizer. \n\
                      A is a M*N real matrix (M<N), x is sparse. A good regularizer is an L1 regularizer. \n\
                      We first split the operator into 0.5*||Ax-b||^2, f(x), and lambda*||x||_1, g(x), where lambda is user specified weight. \n\
                      Then we turn constraint into a barrier term, -mu*sum(log(x)), calling it h(x). \n\
                      Then, we have three operator problem description, F(x) = f(x) + g(x) + h(x). \n\
                      We then solve the problem F(x) with three options: adaptive scalar weight, and Barzilai-Borwein diagonal variable metric. \n";

typedef struct {
  PetscInt  M,N,K,reg;
  PetscReal lambda,eps,mumin;
  PetscReal mu;
  Mat       A,ATA;
  Vec       c,xlb,xub,x,b,workM,workN,workN2,workN3,xGT;    /* observation b, ground truth xGT, the lower bound and upper bound of x*/
} AppCtx;

PetscErrorCode Monitor(Tao tao, void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      v1,v2;
  AppCtx         *user = (AppCtx*)ctx;

  ierr = VecCopy(user->x,user->workN);CHKERRQ(ierr);
  ierr = VecAXPY(user->workN,-1,user->xGT);CHKERRQ(ierr);
  ierr = VecNorm(user->workN,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(user->xGT,NORM_2,&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2));CHKERRQ(ierr);
  return(0);
}

/* Prox solver for g(x), lambda*|x|_1
   out = x - H^-1*lbd*e                */
static PetscErrorCode GProx(Tao tao)
{
  PetscErrorCode ierr;
  PetscReal      lambda;
  AppCtx         *user;
  Vec            out,work,work2;
  Mat            vm;

  PetscFunctionBegin;
  user = NULL;
  ierr = TaoShellGetContext(tao, (void**) &user);CHKERRQ(ierr);

  work   = user->workN;
  work2  = user->workN2;
  lambda = user->lambda;

  ierr = VecSet(work,1.);CHKERRQ(ierr);
  ierr = TaoGetSolutionVector(tao, &out);CHKERRQ(ierr);
  ierr = TaoVMNOSGetVMMat(tao,&vm);CHKERRQ(ierr);

  ierr = MatMult(vm,work,work2);CHKERRQ(ierr);
  ierr = VecScale(work2,lambda);CHKERRQ(ierr);
  ierr = VecAXPY(out,-1.,work2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Prox solver for H(x), -mu*sum(log(x_i))
 * out = (x + sqrt(x**2 +4*mu*H^(-1))) / 2 */
static PetscErrorCode HProx(Tao tao)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  Vec            out,work,work2;
  Mat            vm;

  PetscFunctionBegin;
  user = NULL;
  ierr = TaoShellGetContext(tao, (void**) &user);CHKERRQ(ierr);

  work  = user->workN;
  work2 = user->workN2;
  ierr  = VecSet(work,1.);CHKERRQ(ierr);

  ierr = TaoGetSolutionVector(tao, &out);CHKERRQ(ierr);
  ierr = TaoVMNOSGetVMMat(tao,&vm);CHKERRQ(ierr);

  ierr = MatMult(vm,work,work2);CHKERRQ(ierr);
  ierr = VecScale(work2,4*(user->mu));CHKERRQ(ierr);
  ierr = VecPointwiseMult(work,out,out);CHKERRQ(ierr);
  ierr = VecAXPY(work,1.,work2);CHKERRQ(ierr);
  ierr = VecSqrtAbs(work);CHKERRQ(ierr);
  ierr = VecAXPY(out,1.,work);CHKERRQ(ierr);
  ierr = VecScale(out,0.5);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode BarrierUpdate(Tao tao, PetscInt iter, void *ctx)
{
  AppCtx             *user = (AppCtx*)ctx;
  PetscInt           its;
  PetscReal          f,gnorm,cnorm,xdiff;
  PetscReal          gatol,grtol,gttol;
  TaoConvergedReason reason;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TaoGetSolutionStatus(tao, &its, &f, &gnorm, &cnorm, &xdiff, &reason);CHKERRQ(ierr);
  ierr = TaoGetTolerances(tao,&gatol,&grtol,&gttol);CHKERRQ(ierr);
  if (gnorm < 1.e-6 && (user->mu > 1.e-18)) user->mu =  user->mu / 1.1;
  else if (user->mu < 1.e-18) {
    ierr = TaoSetConvergedReason(tao,TAO_CONVERGED_USER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode F1Objective(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 */
  ierr = MatMult(user->A,X,user->workM);CHKERRQ(ierr);
  ierr = VecAXPY(user->workM,-1,user->b);CHKERRQ(ierr);
  ierr = VecDot(user->workM,user->workM,f);CHKERRQ(ierr);
  *f  *= 0.5;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode F1Gradient(Tao tao,Vec X,Vec g,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Gradient. ATAx-ATb */
  ierr = MatMult(user->ATA,X,user->workN);CHKERRQ(ierr);
  ierr = MatMultTranspose(user->A,user->b,user->workN2);CHKERRQ(ierr);
  ierr = VecWAXPY(g,-1.,user->workN2,user->workN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode FullObj(Tao tao,Vec X,PetscReal *f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscReal      f_reg;

  PetscFunctionBegin;
  /* Objective  0.5*||Ax-b||_2^2 + lambda*||x||_1 */
  ierr = MatMult(user->A,X,user->workM);CHKERRQ(ierr);
  ierr = VecAXPY(user->workM,-1,user->b);CHKERRQ(ierr);
  ierr = VecDot(user->workM,user->workM,f);CHKERRQ(ierr);
  *f  *= 0.5;

  ierr = VecNorm(X,NORM_1,&f_reg);CHKERRQ(ierr);
  ierr = VecCopy(X,user->workN);CHKERRQ(ierr);
  *f  += user->lambda*f_reg;
  PetscFunctionReturn(0);
}

PetscErrorCode InitializeUserData(AppCtx *user)
{
  char           dataFile[] = "tomographyData_A_b_xGT";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. Debug: change from "tomographyData_A_b_xGT" to "cs1Data_A_b_xGT". */
  PetscViewer    fd;   /* used to load data from file */
  PetscErrorCode ierr;
  PetscFunctionBegin;

  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->A);CHKERRQ(ierr);
  ierr = MatSetType(user->A,MATAIJ);CHKERRQ(ierr);
  ierr = MatLoad(user->A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->b);CHKERRQ(ierr);
  ierr = VecLoad(user->b,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->xGT);CHKERRQ(ierr);
  ierr = VecLoad(user->xGT,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = MatGetSize(user->A,&user->M,&user->N);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&(user->x));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workM));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workN));CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&(user->workN2));CHKERRQ(ierr);
  ierr = VecSetSizes(user->x,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workM,PETSC_DECIDE,user->M);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workN,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetSizes(user->workN2,PETSC_DECIDE,user->N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workM);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workN);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->workN2);CHKERRQ(ierr);

  ierr = VecDuplicate(user->workN,&(user->workN3));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->xlb));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->xub));CHKERRQ(ierr);
  ierr = VecDuplicate(user->x,&(user->c));CHKERRQ(ierr);
  ierr = VecSet(user->xlb,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->c,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->xub,PETSC_INFINITY);CHKERRQ(ierr);

  ierr = MatTransposeMatMult(user->A,user->A, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &(user->ATA));CHKERRQ(ierr);

  ierr = MatAssemblyBegin(user->ATA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->ATA,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  user->lambda = 1.e-8;
  user->mu     = 1.e-6;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Configure separable objection example", "tomovmnos.c");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda", "The regularization multiplier. 1 default", "tomovmnos.c", user->lambda, &(user->lambda), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "Starting value for barrier term for constatint.", "tomovmnos.c", user->mu, &(user->mu), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&user->A);CHKERRQ(ierr);
  ierr = MatDestroy(&user->ATA);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xGT);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xlb);CHKERRQ(ierr);
  ierr = VecDestroy(&user->xub);CHKERRQ(ierr);
  ierr = VecDestroy(&user->b);CHKERRQ(ierr);
  ierr = VecDestroy(&user->x);CHKERRQ(ierr);
  ierr = VecDestroy(&user->c);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN3);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN2);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workN);CHKERRQ(ierr);
  ierr = VecDestroy(&user->workM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Tao            tao,taoarr[2];
  PetscReal      v1,v2;
  AppCtx*        user;
  PetscViewer    fd;
  char           resultFile[] = "tomographyResult_x";

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ierr = InitializeUserData(user);CHKERRQ(ierr);

  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao, TAOVMNOS);CHKERRQ(ierr);
  ierr = TaoVMNOSSetSeparableOperatorCount(tao,2);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao, user->x);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSetUp(tao);CHKERRQ(ierr);
  ierr = TaoSetUpdate(tao,BarrierUpdate,(void*)user);CHKERRQ(ierr); /* Reduces barrier term mu */

  ierr = TaoSetObjectiveRoutine(tao, FullObj, (void*)user);CHKERRQ(ierr);
  ierr = TaoVMNOSSetF1ObjectiveRoutine(tao, F1Objective, (void*)user);CHKERRQ(ierr);
  ierr = TaoVMNOSSetF1GradientRoutine(tao, F1Gradient, (void*)user);CHKERRQ(ierr);

  ierr = TaoVMNOSGetSubsolvers(tao,taoarr);

  ierr = TaoSetType(taoarr[1],TAOSHELL);CHKERRQ(ierr);
  ierr = TaoShellSetContext(taoarr[1], (void*) user);CHKERRQ(ierr);
  ierr = TaoShellSetSolve(taoarr[1], GProx);CHKERRQ(ierr);

  ierr = TaoSetType(taoarr[0],TAOSHELL);CHKERRQ(ierr);
  ierr = TaoShellSetContext(taoarr[0], (void*) user);CHKERRQ(ierr);
  ierr = TaoShellSetSolve(taoarr[0], HProx);CHKERRQ(ierr);

  ierr = TaoSetMonitor(tao,Monitor,(void*) user,NULL);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,resultFile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = VecView(user->x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* compute the error */
  ierr = VecAXPY(user->x,-1,user->xGT);CHKERRQ(ierr);
  ierr = VecNorm(user->x,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(user->xGT,NORM_2,&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2));CHKERRQ(ierr);

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = DestroyContext(user);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single !__float128 !define(PETSC_USE_64BIT_INDICES)

   test:
      suffix: 1
      localrunfiles: tomographyData_A_b_xGT
      args:  -tao_vmnos_stepsize 13038.2 -lambda 1.e-8 -tao_smonitor -tao_vmnos_vm_update adaptive -tao_gatol 1.e-8 -tao_max_it 1000

   test:
      suffix: 2
      localrunfiles: tomographyData_A_b_xGT
      args:  -tao_vmnos_stepsize 13038.2 -lambda 1.e-8 -tao_smonitor -tao_vmnos_vm_update bb -tao_gatol 1.e-8 -tao_max_it 1000

TEST*/
