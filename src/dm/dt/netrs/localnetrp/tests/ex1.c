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
  Mat TrafficDistribution,Jacineq; 
  PetscPointFlux flux; 
  PetscReal sigma,fluxsigma; /* point of the maximum of the flux, assumes flux is concave */
  Vec       lowerbounds, upperbounds; /* upper bounds are dependant on the inputs to the Riemann Problem */
  Vec       U,FluxU,Gradient; /* Riemann Data for problem */
  Vec       GammaMax; /* Maximum Flux that can be obtained on a road. Used in the A * FluxStar <= GammaMax term */
  PetscInt  numedges,numinedges; /* topology of the network */
  Vec       CI; /* Vector holding the inequality constraints. Odd that TAO seems to require the user to manage this */
  PetscBool *edgein ,initview; 
} AppCtx;

/* -------- User-defined Routines --------- */
PetscErrorCode InitializeProblem(AppCtx *);
PetscErrorCode DestroyProblem(AppCtx *);
PetscErrorCode FormObjective(Tao, Vec, PetscReal *, void *);
PetscErrorCode FormObjectiveGradient(Tao, Vec, Vec, void *);
PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode FormInequalityConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormInequalityJacobian(Tao, Vec, Mat, Mat, void *);

PetscErrorCode main(int argc, char **argv)
{
  Tao         tao;
  AppCtx      user; /* application context */
  Vec         fluxStar, G, CI; 
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
  PetscCall(TaoSetJacobianInequalityRoutine(tao, user.Jacineq, user.Jacineq, FormInequalityJacobian, (void *)&user));

  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPDIPM, &pdipm));
  PetscCheck(!pdipm, PetscObjectComm((PetscObject)tao), PETSC_ERR_SUP, "Cannot use PDIPM method in its current form. Requires a special Hessian Matrix");

  /* Print out an initial view of the problem */
  if (user.initview) {
    PetscCall(TaoSetUp(tao));
    PetscCall(VecDuplicate(user.FluxU, &G));
    PetscCall(FormFunctionGradient(tao, user.FluxU, &f, G, (void *)&user));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial point X:\n"));
    PetscCall(VecView(user.FluxU, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial objective f(x) = %g\n", (double)f));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial gradient \n"));
    PetscCall(VecView(G, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&G));
    PetscCall(FormInequalityJacobian(tao, user.FluxU, user.Jacineq, user.Jacineq, (void *)&user));
    PetscCall(MatCreateVecs(user.TrafficDistribution, NULL, &CI));
    PetscCall(FormInequalityConstraints(tao, user.FluxU, CI, (void *)&user));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nInitial inequality constraints and Jacobian:\n"));
    PetscCall(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatView(user.Jacineq, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecDestroy(&CI));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(TaoSolve(tao));
  PetscCall(TaoGetSolution(tao, &fluxStar));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Found solution:\n"));
  PetscCall(VecView(fluxStar, PETSC_VIEWER_STDOUT_WORLD));
  //PetscCall(MatCreateVecs(user.TrafficDistribution, NULL, &CI));
  //PetscCall(TaoComputeInequalityConstraints(tao,fluxStar,CI)); 
  //PetscCall(VecView(CI, PETSC_VIEWER_STDOUT_WORLD));
  //PetscCall(MatDuplicate(user.TrafficDistribution,MAT_DO_NOT_COPY_VALUES,&user.Jacineq)); 
  //PetscCall(TaoComputeJacobianInequality(tao,fluxStar,user.Jacineq,NULL)); 
  //PetscCall(VecDestroy(&CI));

  //PetscCall(MatView(user.TrafficDistribution,PETSC_VIEWER_STDOUT_WORLD)); 

  /* Free objects */
  PetscCall(DestroyProblem(&user));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
}

/* basic LWR flux, 
  Normalized so that flux(\sigma) = 1.0, where \sigma is st flux(\sigma) = \max_{u} \flux(u). 

  \sigma = 1/2, by simple optimization. 

  Note that for 
  \rho = \frac{\sqrt{2} \pm 1}{2\sqrt{2}}
  we have 
  flux(\rho) = 1/2 . 
 */
void FluxFunction(void *ctx, const PetscReal *u, PetscReal *flux) 
{
  flux[0] = 4.0 *u[0] * (1.0- u[0]); 
}
PetscErrorCode InitializeProblem(AppCtx *user)
{
  PetscMPIInt size;
  PetscInt    i,j,k; 
  PetscScalar *u,*fluxu,*upperbnd,*gammamax; 

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  user->initview = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-init_view", &user->initview, NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-numedges",&user->numedges,NULL));
  user->numedges = 4; /* hard set for now */
  user->flux = FluxFunction; 
  user->sigma = 0.5; 
  user->fluxsigma = 1.0; 

  /* create vector x and set initial values */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user->U));
  PetscCall(VecSetSizes(user->U, PETSC_DECIDE, user->numedges));
  PetscCall(VecSetFromOptions(user->U));
  PetscCall(VecGetArray(user->U,&u)); 

  PetscCall(PetscMalloc1(user->numedges,&user->edgein)); 
  user->numinedges = 0; 
  for(i=0; i<user->numedges; i++) {
    if(i < PetscCeilInt(user->numedges,2)) {
      user->edgein[i] = PETSC_TRUE; 
      u[i] = 1.0; 
    } else {
      user->edgein[i] = PETSC_FALSE; 
      u[i] = 0.0; 
      user->numinedges++; 
    }
  }

  // buggy thing to do here..... 

  /* values for example in benedettos book */

  u[0] = (PetscSqrtScalar(2.0) - 1)/(2* PetscSqrtScalar(2)); 
  u[1] = 3./4.; 
  u[2] = 1./4.; 
  u[3] = (PetscSqrtScalar(2.0) + 1)/(2* PetscSqrtScalar(2)); 

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
  PetscCall(MatSetValue(user->TrafficDistribution,0,0,1./3.,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,0,1,1./4.,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,1,0,2./3.,INSERT_VALUES)); 
  PetscCall(MatSetValue(user->TrafficDistribution,1,1,3./4.,INSERT_VALUES)); 
  PetscCall(MatAssemblyBegin(user->TrafficDistribution,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->TrafficDistribution,MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(user->TrafficDistribution,-1)); /* negative as things are flipped for the TAO form */

  /* Create Vector for the Inequality Constraints. Tao does not create 
  them automatically, for some reason. Matrix for the Jacobian is the Traffic Distribution Matrix (affine operator) */
  PetscCall(VecDuplicate(user->GammaMax, &user->CI));
  PetscCall(MatDuplicate(user->TrafficDistribution,MAT_DO_NOT_COPY_VALUES,&user->Jacineq)); 
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
  E(\mathbf{\gamma}) = -\sum_{e into v} \gamma_e 
  and \nabla E  = -\mathbf{1}. 
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
  const PetscScalar *x;
  MPI_Comm           comm;
  PetscMPIInt        size;
  PetscInt          i, n; 
  PetscScalar       *g; 
 

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) tao, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCheck(size == 1, comm, PETSC_ERR_USER_INPUT,"This Callback requires sequential communicator. Communicator for TAO object had size %"PetscInt_FMT, size);

  PetscCall(VecGetSize(X,&n)); 
  PetscCall(VecGetArrayRead(X, &x));
  *f = 0; 
  for(i=0; i<n; i++) *f -= x[i]; 
  PetscCall(VecRestoreArrayRead(X, &x));

  PetscCall(VecGetSize(G,&n)); 
  PetscCall(VecGetArray(G,&g));
  for(i=0; i<n; i++ ) g[i] = -1.0; 
  PetscCall(VecRestoreArray(G,&g)); 
  PetscFunctionReturn(0);
}

/* Evaluate
  E(\mathbf{\gamma}) = \sum_{e into v} -\gamma_e 
*/
PetscErrorCode FormObjective(Tao tao, Vec X, PetscReal *f,  void *ctx)
{
  const PetscScalar *x;
  MPI_Comm           comm;
  PetscMPIInt        size;
  PetscInt          i, n; 
 

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) tao, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCheck(size == 1, comm, PETSC_ERR_USER_INPUT,"This Callback requires sequential communicator. Communicator for TAO object had size %"PetscInt_FMT, size);

  PetscCall(VecGetSize(X,&n)); 
  PetscCall(VecGetArrayRead(X, &x));
  *f = 0; 
  for(i=0; i<n; i++) *f -= x[i]; 
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}
/* Evaluate \nabla E  = -\mathbf{1} where 
  E(\mathbf{\gamma}) = -\sum_{e into v} \gamma_e . 
*/
PetscErrorCode FormObjectiveGradient(Tao tao, Vec X, Vec G, void *ctx)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  PetscInt          i, n; 
  PetscScalar       *g; 
 

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) tao, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCheck(size == 1, comm, PETSC_ERR_USER_INPUT,"This Callback requires sequential communicator. Communicator for TAO object had size %"PetscInt_FMT, size);

  PetscCall(VecGetSize(G,&n)); 
  PetscCall(VecGetArray(G,&g));
  for(i=0; i<n; i++) g[i] = -1.0; 
  PetscCall(VecRestoreArray(G,&g)); 
  PetscFunctionReturn(0);
}

/* Evaluate
  h(x) >= 0 where 
  h(x)  = GammaMax + A * x 

  where A is the negative of the usual Traffic Distribution Matrix as shown below. 

  In traffic netowrk papers this is the condition that 
  A * \mathbf{\gamma} \in \Omega_{n+1} \times \hdots \times \Omega_{n+m}, i.e 

 0 \geq A * \mathbf{\gamma} \leq [\gamma_j^{max}(\rho_{j,0}) : j = n+1, \hdots , n+m] = GammaMax. 

 The lower bound is automatically satisified by the constraints on \gamma. 
*/
PetscErrorCode FormInequalityConstraints(Tao tao, Vec X, Vec CI, void *ctx)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  AppCtx            *user = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)X, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_USER_INPUT,"This Callback requires sequential communicator. Communicator for TAO object had size %"PetscInt_FMT, size);
  PetscCall(MatMultAdd(user->TrafficDistribution,X,user->GammaMax,CI)); 
  PetscFunctionReturn(0);
}

/*
  grad h = A, which is already formed 
*/
PetscErrorCode FormInequalityJacobian(Tao tao, Vec X, Mat JI, Mat JIpre, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;

  PetscFunctionBegin;
  PetscCall(MatCopy(user->TrafficDistribution,JI,SAME_NONZERO_PATTERN)); 
  if(JI != JIpre) {
    PetscCall(MatCopy(user->TrafficDistribution,JIpre,SAME_NONZERO_PATTERN)); 
  }
  PetscFunctionReturn(0);
}