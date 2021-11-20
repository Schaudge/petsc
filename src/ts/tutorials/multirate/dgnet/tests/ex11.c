static const char help[] = "DGNET Coupling Test Example";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  ierr = DGNetworkMonitorView(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 static PetscErrorCode MakeOrder(PetscInt dof, PetscInt *order,PetscInt maxdegree)
{
  PetscInt  i; 
  for(i=0; i<dof; i++) order[i] = maxdegree;
  PetscFunctionReturn(0);
}
int main(int argc,char *argv[])
{
  char              physname[256] = "shallow", errorestimator[256] = "lax" ;
  PetscFunctionList physics = 0,errest = 0; 
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         dgnet;
  PetscInt          maxorder=1;
  PetscReal         maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscBool         limit=PETSC_TRUE;
  DGNetworkMonitor  monitor=NULL;
  NRSErrorEstimator errorest; 

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);



    /* register error estimator functions */
    ierr = PetscFunctionListAdd(&errest,"roe"         ,NetRSRoeErrorEstimate);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&errest,"lax"         ,NetRSLaxErrorEstimate);CHKERRQ(ierr);
    ierr = PetscFunctionListAdd(&errest,"taylor"      ,NetRSTaylorErrorEstimate);CHKERRQ(ierr);

    ierr = PetscCalloc1(1,&dgnet);CHKERRQ(ierr); /* Replace with proper dgnet creation function */
    /* Set default values */
    dgnet->comm           = comm;
    dgnet->cfl            = 0.9;
    dgnet->networktype    = 6;
    dgnet->hratio         = 1;
    maxtime               = 2.0;
    dgnet->Mx             = 10;
    dgnet->initial        = 1;
    dgnet->ndaughters     = 2;
    dgnet->length         = 3.0;
    dgnet->view           = PETSC_FALSE;
    dgnet->jumptol        = 0.5;
    dgnet->laxcurve       = PETSC_FALSE;  
    dgnet->diagnosticlow  = 0.5; 
    dgnet->diagnosticup   = 1e-4; 
    dgnet->adaptivecouple = PETSC_FALSE;
    dgnet->linearcoupling = PETSC_TRUE;
    /* Command Line Options */
    ierr = PetscOptionsBegin(comm,NULL,"DGNetwork solver options","");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-errest","","",errest,errorestimator,errorestimator,sizeof(physname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",dgnet->initial,&dgnet->initial,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",dgnet->networktype,&dgnet->networktype,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",dgnet->cfl,&dgnet->cfl,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-length","Length of Edges in the Network","",dgnet->length,&dgnet->length,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",dgnet->Mx,&dgnet->Mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ndaughters","Number of daughter branches for network type 3","",dgnet->ndaughters,&dgnet->ndaughters,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view","View the DG solution","",dgnet->view,&dgnet->view,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL);CHKERRQ(ierr);    ierr = PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-adaptivecouple","Use adaptive Coupling for Netrs","",dgnet->adaptivecouple,&dgnet->adaptivecouple,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-lax","Use lax curve diagnostic for coupling","",dgnet->laxcurve,&dgnet->laxcurve,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-jumptol","Set jump tolerance for lame one-sided limiter","",dgnet->jumptol,&dgnet->jumptol,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-lincouple","Use lax curve diagnostic for coupling","",dgnet->linearcoupling,&dgnet->linearcoupling,NULL);CHKERRQ(ierr);

    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    /* Choose the physics from the list of registered models */
    {
      PetscErrorCode (*r)(DGNetwork);
      ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
      if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
      /* Create the physics, will set the number of fields and their names */
      ierr = (*r)(dgnet);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(dgnet->physics.dof,&dgnet->physics.order);CHKERRQ(ierr); /* should be constructed by physics */
    ierr = MakeOrder(dgnet->physics.dof,dgnet->physics.order,maxorder);CHKERRQ(ierr);
  
    /* Generate Network Data */
    ierr = DGNetworkCreate(dgnet,dgnet->networktype,dgnet->Mx);CHKERRQ(ierr);
    /* Create DMNetwork */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dgnet->network);CHKERRQ(ierr);
    
    /* Set Network Data into the DMNetwork (on proc[0]) */
    ierr = DGNetworkSetComponents(dgnet);CHKERRQ(ierr);
    /* Delete unneeded data in dgnet */
    ierr = DGNetworkCleanUp(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkBuildTabulation(dgnet);CHKERRQ(ierr);
    if (size == 1 && dgnet->view) {
      ierr = DGNetworkMonitorCreate(dgnet,&monitor);CHKERRQ(ierr);
      ierr = DGNetworkAddMonitortoEdges(dgnet,monitor);CHKERRQ(ierr);
    }

    /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to 
       set all the physics parts at once) */

    ierr = RiemannSolverCreate(dgnet->comm,&dgnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(dgnet->physics.rs,dgnet->physics.user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(dgnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(dgnet->physics.rs,dgnet->physics.fluxeig);CHKERRQ(ierr);
    ierr = RiemannSolverSetRoeAvgFunct(dgnet->physics.rs,dgnet->physics.roeavg);CHKERRQ(ierr);
    ierr = RiemannSolverSetRoeMatrixFunct(dgnet->physics.rs,dgnet->physics.roemat);CHKERRQ(ierr);
    ierr = RiemannSolverSetEigBasis(dgnet->physics.rs,dgnet->physics.eigbasis);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(dgnet->physics.rs,1,dgnet->physics.dof,dgnet->physics.flux2);CHKERRQ(ierr);
    ierr = RiemannSolverSetLaxCurve(dgnet->physics.rs,dgnet->physics.laxcurve);CHKERRQ(ierr);
    ierr = RiemannSolverSetUp(dgnet->physics.rs);CHKERRQ(ierr);

    /* Create Vectors */
    ierr = DGNetworkCreateVectors(dgnet);CHKERRQ(ierr);
    /* Set up component dynamic data structures */
    ierr = DGNetworkBuildDynamic(dgnet);CHKERRQ(ierr);
    /* Set up NetRS */
    ierr = PetscFunctionListFind(errest,errorestimator,&errorest);CHKERRQ(ierr);
    ierr = DGNetworkAssignNetRS(dgnet,dgnet->physics.rs,errorest,1);CHKERRQ(ierr);
    ierr = DGNetworkProject(dgnet,dgnet->X,0.0);CHKERRQ(ierr);

  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,dgnet);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(ts,NULL,DGNetRHS_NETRSVERSION,dgnet);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dgnet->cfl/dgnet->Mx/(2*maxorder+1));CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
  if (size == 1 && dgnet->view) {
    ierr = TSMonitorSet(ts, TSDGNetworkMonitor,monitor, NULL);CHKERRQ(ierr);
  }
  if (limit) {
      /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */  
    ierr = DGNetlimiter(ts,0,0,&dgnet->X);CHKERRQ(ierr);
    ierr = TSSetPostStage(ts,DGNetlimiter);CHKERRQ(ierr);
  } 

  /* Clean up the output directory (this output data should be redone */

    ierr = PetscRMTree("output");CHKERRQ(ierr);
    ierr = PetscMkdir("output");CHKERRQ(ierr);
  ierr = TSSolve(ts,dgnet->X);CHKERRQ(ierr);

 /* Clean up */
  if(dgnet->view && size==1) {
      ierr = DGNetworkMonitorDestroy(&monitor);CHKERRQ(ierr);

  }

    ierr = RiemannSolverDestroy(&dgnet->physics.rs);CHKERRQ(ierr);
    ierr = PetscFree(dgnet->physics.order);CHKERRQ(ierr);
    ierr = DGNetworkDestroyNetRS(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkDestroy(dgnet);CHKERRQ(ierr); /* Destroy all data within the network and within dgnet */
    ierr = DMDestroy(&dgnet->network);CHKERRQ(ierr);
    ierr = PetscFree(dgnet);CHKERRQ(ierr);

  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}