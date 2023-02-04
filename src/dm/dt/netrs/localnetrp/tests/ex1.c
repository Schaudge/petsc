/* 
 Quick test problem for setting up the TAO optimization problem for the traffic coupling conditions, solved using allm

 Though it really is a linear programming problme and should be using something else (I think....)

 Scalar only, as it doesn't need parallel stuff. 
*/


#include <petsctao.h>
#include <petscnetrp.h>
#include <petscriemannsolver.h>

static char help[] = "TODO \n";

/*
   User-defined application context - contains data needed by the application
*/
typedef struct {
  Mat TrafficDistribution, JacIneq, JacFun; /* odd that JacIneq and JacFun are required to be held by the user */
  PetscPointFlux flux; 
  PetscPointFluxDerVec Gradflux; 
  PetscReal sigma,fluxsigma; /* point of the maximum of the flux, assumes flux is concave */
  Vec       lowerbounds, upperbounds; /* upper bounds are dependant on the inputs to the Riemann Problem */
  Vec       U,FluxU; /* Riemann Data for problem */
  Vec       GammaMax; /* Maximum Flux that can be obtained on a road. Used in the A * FluxStar <= GammaMax term */
  PetscInt  numedges,numinedges; /* topology of the network */
  Vec       CI; /* Vector holding the inequality constraints. Odd that TAO seems to require the user to manage this */
  PetscBool *edgein ,initview; 
} AppCtx;

/* -------- User-defined Routines --------- */
PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormObjective(Tao, Vec, PetscReal *, void *);
PetscErrorCode FormObjectiveGradient(Tao, Vec, Vec, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormInequalityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormEqualityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormInequalityJacobian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode FormEqualityJacobian(Tao, Vec, Mat, Mat, void *);

PetscErrorCode main(int argc, char **argv)
{
  Tao         tao;
  KSP         ksp;
  PC          pc;
  AppCtx      user; /* application context */
  Vec         fluxStar, G, CI, CE; 
  PetscMPIInt size;
  TaoType     type;
  PetscReal   f;
  PetscBool   pdipm; 

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "More than 1 processor detected. Example written to use max of 1 processor.");

  PetscCall(InitializeProblem(&user)); /* sets up problem, function below */

  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOALMM));
  PetscCall(TaoSetSolution(tao, user.FluxU));
  PetscCall(TaoSetVariableBounds(tao, user.lowerbounds, user.upperbounds));
  PetscCall(TaoSetObjective(tao,FormObjective,(void *)&user)); 
  PetscCall(TaoSetGradient(tao,NULL,FormObjectiveGradient,(void *)&user));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));
  PetscCall(TaoSetTolerances(tao, 1.e-4, 0.0, 0.0));
  PetscCall(TaoSetConstraintTolerances(tao, 1.e-4, 0.0));
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TaoSetInequalityConstraintsRoutine(tao, user.CI, FormInequalityConstraints, (void *)&user));
  PetscCall(TaoSetJacobianInequalityRoutine(tao, user.JacIneq, user.JacIneq, FormInequalityJacobian, (void *)&user));

  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPDIPM, &pdipm));
  PetscCheck(!pdipm, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Cannot use PDIPM method in its current form. Requires a special Hessian Matrix");

  /* Print out an initial view of the problem */
  if (user.initview) {
    PetscCall(TaoSetUp(tao));
    PetscCall(VecDuplicate(user.U, &G));
    PetscCall(FormFunctionGradient(tao, user.FluxU, &f, G, (void *)&user));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial point X:\n"));
    PetscCall(VecView(user.FluxU, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial objective f(x) = %g\n", (double)f));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial gradient \n"));
    PetscCall(VecView(G, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&G));
    PetscCall(FormInequalityJacobian(tao, user.FluxU, user.JacIneq, user.JacIneq, (void *)&user));
    PetscCall(MatCreateVecs(user.JacIneq, NULL, &CI));
    PetscCall(FormInequalityConstraints(tao, user.FluxU, CI, (void *)&user));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial inequality constraints and Jacobian:\n"));
    PetscCall(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatView(user.JacIneq, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&CI));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolution(tao, &fluxStar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Found solution:\n"));
  PetscCall(VecView(fluxStar, PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  PetscCall(DestroyProblem(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscMPIInt size;
  PetscMPIInt rank;
  PetscInt    nloc, neloc, niloc,i,j,k; 
  PetscScalar *u,*fluxu,*upperbnd,*gammamax; 

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  user->initview = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-init_view", &user->initview, NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-numedges",&user->numedges,NULL));
  user->numedges = 2.0; /* hard set for now */

  /* create vector x and set initial values */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->U));
  PetscCall(VecSetSizes(user->U, PETSC_DECIDE, user->numedges));
  PetscCall(VecSetFromOptions(user->U));
  PetscCall(VecGetArray(user->U,&u)); 

  PetscCall(PetscMalloc1(user->numedges,&user->edgein)); 
  user->numinedges = 0; 
  for(i=0; i<user->numedges; i++) {
    if(i < PetscCeilInt(user->numedges,2)) {
      user->edgein[i] = PETSC_FALSE; 
      u[i] = 0; 
    } else {
      user->edgein[i] = PETSC_TRUE; 
      u[i] = 1.0; 
      user->numinedges++; 
    }
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->FluxU));
  PetscCall(VecSetSizes(user->FluxU, PETSC_DECIDE, user->numinedges));
  PetscCall(VecSetFromOptions(user->FluxU));
  PetscCall(VecGetArray(user->FluxU,&fluxu)); 
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->GammaMax)); 
  PetscCall(VecSetSizes(user->GammaMax, PETSC_DECIDE,user->numedges-user->numinedges));
  PetscCall(VecSetFromOptions(user->GammaMax));
    /* create and set lower and upper bound vectors */
  PetscCall(VecDuplicate(user->FluxU, &user->lowerbounds));
  PetscCall(VecDuplicate(user->FluxU, &user->upperbounds));
  PetscCall(VecSet(user->lowerbounds, 0.0));

  PetscCall(VecGetArray(user->upperbounds,&upperbnd));
  PetscCall(VecGetArray(user->GammaMax,&gammamax));

  j=0; 
  k=0; 
  for(i=0; i<user->numedges; i++) {
    if(user->edgein[i] == PETSC_TRUE ) {
      user->flux(NULL,&u[i],&fluxu[j]);
      upperbnd[j] = (u[i] <= user->sigma ) ? fluxu[j] : user->fluxsigma; 
      j++; 
    } else {
      if (u[i]> user->sigma) {
        user->flux(NULL,&u[i],&gammamax[k]);
      } else {
        gammamax[k] = user->fluxsigma;
      }
      k++; 
    }
  }

  PetscCall(VecRestoreArray(user->GammaMax,&gammamax));
  PetscCall(VecRestoreArray(user->U,&u));
  PetscCall(VecRestoreArray(user->FluxU,&fluxu));
  PetscCall(VecRestoreArray(user->upperbounds,&upperbnd));


  /* Create Traffic Distribution Matrix */
  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,2,2,NULL,&user->TrafficDistribution)); 
  PetscCall(MatSetUp(user->TrafficDistribution));
  PetscCall(MatSetValue(user->TrafficDistribution,0,0,1/3,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,0,1,1/4,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,1,0,2/3,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,1,1,3/4,INSERT_VALUES)); 
  PetscCall(MatAssemblyBegin(user->TrafficDistribution,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->TrafficDistribution,MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(user->TrafficDistribution,-1)); /* negative as things are flipped for the TAO form */

  /* Create Vector for the Inequality Constraints. Tao does not create 
  them automatically, for some reason. Matrix for the Jacobian is the Traffic Distribution Matrix (affine operator) */
  PetscCall(VecDuplicate(user->GammaMax, &user->CI));
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyProblem(AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(&user->TrafficDistribution));
  PetscCall(VecDestroy(&user->U));
  PetscCall(VecDestroy(&user->CI));
  PetscCall(VecDestroy(&user->upperbounds));
  PetscCall(VecDestroy(&user->lowerbounds));
  PetscCall(VecDestroy(&user->GammaMax));
  PetscCall(VecDestroy(&user->FluxU));
  PetscFunctionReturn(0);
}

/* Evaluate
   f(x) = (x0 - 2)^2 + (x1 - 2)^2 - 2*(x0 + x1)
   G = grad f = [2*(x0 - 2) - 2;
                 2*(x1 - 2) - 2]
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  PetscScalar        g;
  const PetscScalar *x;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  PetscReal          fin;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  fin = 0.0;
  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    fin = (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 2.0) * (x[1] - 2.0) - 2.0 * (x[0] + x[1]);
    g   = 2.0 * (x[0] - 2.0) - 2.0;
    PetscCall(VecSetValue(G, 0, g, INSERT_VALUES));
    g = 2.0 * (x[1] - 2.0) - 2.0;
    PetscCall(VecSetValue(G, 1, g, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCallMPI(MPI_Allreduce(&fin, f, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCall(VecAssemblyBegin(G));
  PetscCall(VecAssemblyEnd(G));
  PetscFunctionReturn(0);
}

/* Evaluate
   H = fxx + grad (grad g^T*DI) - grad (grad h^T*DE)]
     = [ 2*(1+de[0]-di[0]+di[1]), 0;
                   0,             2]
*/
PetscErrorCode FormHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  Vec                DE, DI;
  const PetscScalar *de, *di;
  PetscInt           zero = 0, one = 1;
  PetscScalar        two = 2.0;
  PetscScalar        val = 0.0;
  Vec                Deseq, Diseq;
  VecScatter         Descat, Discat;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(TaoGetDualVariables(tao, &DE, &DI));

  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (!user->noeqflag) {
    PetscCall(VecScatterCreateToZero(DE, &Descat, &Deseq));
    PetscCall(VecScatterBegin(Descat, DE, Deseq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(Descat, DE, Deseq, INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscCall(VecScatterCreateToZero(DI, &Discat, &Diseq));
  PetscCall(VecScatterBegin(Discat, DI, Diseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Discat, DI, Diseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    if (!user->noeqflag) { PetscCall(VecGetArrayRead(Deseq, &de)); /* places equality constraint dual into array */ }
    PetscCall(VecGetArrayRead(Diseq, &di)); /* places inequality constraint dual into array */

    if (!user->noeqflag) {
      val = 2.0 * (1 + de[0] - di[0] + di[1]);
      PetscCall(VecRestoreArrayRead(Deseq, &de));
      PetscCall(VecRestoreArrayRead(Diseq, &di));
    } else {
      val = 2.0 * (1 - di[0] + di[1]);
    }
    PetscCall(VecRestoreArrayRead(Diseq, &di));
    PetscCall(MatSetValues(H, 1, &zero, 1, &zero, &val, INSERT_VALUES));
    PetscCall(MatSetValues(H, 1, &one, 1, &one, &two, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  if (!user->noeqflag) {
    PetscCall(VecScatterDestroy(&Descat));
    PetscCall(VecDestroy(&Deseq));
  }
  PetscCall(VecScatterDestroy(&Discat));
  PetscCall(VecDestroy(&Diseq));
  PetscFunctionReturn(0);
}

/* Evaluate
   h = [ x0^2 - x1;
         1 -(x0^2 - x1)]
*/
PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  const PetscScalar *x;
  PetscScalar        ci;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    ci = x[0] * x[0] - x[1];
    PetscCall(VecSetValue(CI, 0, ci, INSERT_VALUES));
    ci = -x[0] * x[0] + x[1] + 1.0;
    PetscCall(VecSetValue(CI, 1, ci, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCall(VecAssemblyBegin(CI));
  PetscCall(VecAssemblyEnd(CI));
  PetscFunctionReturn(0);
}

/* Evaluate
   g = [ x0^2 + x1 - 2]
*/
PetscErrorCode FormEqualityConstraints(Tao tao, Vec X, Vec CE, void *ctx)
{
  const PetscScalar *x;
  PetscScalar        ce;
  MPI_Comm           comm;
  PetscMPIInt        rank;
  AppCtx            *user = (AppCtx *)ctx;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(Xseq, &x));
    ce = x[0] * x[0] + x[1] - 2.0;
    PetscCall(VecSetValue(CE, 0, ce, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(Xseq, &x));
  }
  PetscCall(VecAssemblyBegin(CE));
  PetscCall(VecAssemblyEnd(CE));
  PetscFunctionReturn(0);
}

/*
  grad h = [  2*x0, -1;
             -2*x0,  1]
*/
PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  PetscInt           zero = 0, one = 1, cols[2];
  PetscScalar        vals[2];
  const PetscScalar *x;
  Vec                Xseq = user->Xseq;
  VecScatter         scat = user->scat;
  MPI_Comm           comm;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecScatterBegin(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat, X, Xseq, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecGetArrayRead(Xseq, &x));
  if (rank == 0) {
    cols[0] = 0;
    cols[1] = 1;
    vals[0] = 2 * x[0];
    vals[1] = -1.0;
    PetscCall(MatSetValues(JI, 1, &zero, 2, cols, vals, INSERT_VALUES));
    vals[0] = -2 * x[0];
    vals[1] = 1.0;
    PetscCall(MatSetValues(JI, 1, &one, 2, cols, vals, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(Xseq, &x));
  PetscCall(MatAssemblyBegin(JI, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JI, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*
  grad g = [2*x0
             1.0 ]
*/
PetscErrorCode FormEqualityJacobian(Tao tao, Vec X, Mat JE, Mat JEpre, void *ctx)
{
  PetscInt           zero = 0, cols[2];
  PetscScalar        vals[2];
  const PetscScalar *x;
  PetscMPIInt        rank;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)tao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank == 0) {
    PetscCall(VecGetArrayRead(X, &x));
    cols[0] = 0;
    cols[1] = 1;
    vals[0] = 2 * x[0];
    vals[1] = 1.0;
    PetscCall(MatSetValues(JE, 1, &zero, 2, cols, vals, INSERT_VALUES));
    PetscCall(VecRestoreArrayRead(X, &x));
  }
  PetscCall(MatAssemblyBegin(JE, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(JE, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !defined(PETSC_USE_CXX)

   test:
      args: -tao_converged_reason -tao_gatol 1.e-6 -tao_type pdipm -tao_pdipm_kkt_shift_pd
      requires: mumps
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 2
      args: -tao_converged_reason
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 3
      args: -tao_converged_reason -no_eq
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 4
      args: -tao_converged_reason -tao_almm_type classic
      requires: !single
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 5
      args: -tao_converged_reason -tao_almm_type classic -no_eq
      requires: !single
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 6
      args: -tao_converged_reason -tao_almm_subsolver_tao_type bqnktr
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 7
      args: -tao_converged_reason -tao_almm_subsolver_tao_type bncg
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 8
      nsize: 2
      args: -tao_converged_reason
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

   test:
      suffix: 9
      nsize: 2
      args: -tao_converged_reason -vec_type cuda -mat_type aijcusparse
      requires: cuda
      filter: sed  -e "s/CONVERGED_GATOL iterations *[0-9]\{1,\}/CONVERGED_GATOL/g"

TEST*/
