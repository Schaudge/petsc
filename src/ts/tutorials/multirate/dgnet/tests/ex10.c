static const char help[] = "Test for NetRS implementations of coupling conditions";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petscriemannsolver.h>
#include "../physics.h"

/* Temporary test as I rework the internals of DGNet and related tools. In
   In particular I need to make a seperate class for network Riemann Solvers and improve viewer routines */

PetscErrorCode TSDGNetworkMonitor(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  ierr = DGNetworkMonitorView(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode TSDGNetworkMonitor_Nest(TS ts, PetscInt step, PetscReal t, Vec x, void *ctx)
{
  PetscErrorCode     ierr;
  DGNetwork_Nest     dgnet_nest = (DGNetwork_Nest)ctx; 
  PetscInt           i;
  Vec                Xsim; 

  PetscFunctionBegin;
  for(i=0; i<dgnet_nest->numsimulations; i++) {
    ierr = VecNestGetSubVec(x,i,&Xsim);CHKERRQ(ierr);
    ierr = DGNetworkMonitorView(dgnet_nest->monitors[i],Xsim);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* poststep function for comparing the two coupling conditions */
PetscErrorCode TSDGNetworkCompare(TS ts)
{
  PetscErrorCode     ierr;
  DGNetwork_Nest     dgnet_nest;
  Vec                Xsim,Xnest,Xdiff; 
  MPI_Comm           comm; 

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&dgnet_nest);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  Xdiff = dgnet_nest->wrk_vec[0];
  ierr = TSGetSolution(ts,&Xnest);CHKERRQ(ierr);
  if(dgnet_nest->numsimulations !=2) {SETERRQ1(comm,PETSC_ERR_SUP,"Only valid for 2 simulations. \n \
  There were %i simulations",dgnet_nest->numsimulations);}
  ierr = VecNestGetSubVec(Xnest,0,&Xsim);CHKERRQ(ierr);
  ierr = VecCopy(Xsim,Xdiff);CHKERRQ(ierr); 
  ierr = VecNestGetSubVec(Xnest,1,&Xsim);CHKERRQ(ierr);
  ierr = VecAXPY(Xdiff,-1.,Xsim);CHKERRQ(ierr);
  if(dgnet_nest->monitors[2]) {ierr = DGNetworkMonitorView(dgnet_nest->monitors[2],Xdiff);CHKERRQ(ierr);}
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
  char              physname[256] = "shallow";
  PetscFunctionList physics = 0;
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         dgnet;
  PetscInt          i,maxorder=1,numsim=2;
  PetscReal         maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscBool         limit=PETSC_TRUE;
  DGNetworkMonitor  monitor=NULL;

  DGNetwork_Nest    dgnet_nest;
  Vec               Diff, XNest,*Xcomp; 

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);

  /* Build the nest structure */ 
  ierr = PetscCalloc1(1,&dgnet_nest);CHKERRQ(ierr);
  dgnet_nest->numsimulations = numsim;
  ierr = PetscCalloc2(numsim,&dgnet_nest->dgnets,numsim+1,&dgnet_nest->monitors);CHKERRQ(ierr);
  ierr = PetscCalloc1(numsim,&Xcomp);CHKERRQ(ierr);
  /* loop the DGNetwork setup
    Note: This is way too long of code for setting things up, need to break things down more 
  */

  for(i=0; i<numsim; i++) {
    ierr = PetscCalloc1(1,&dgnet_nest->dgnets[i]);CHKERRQ(ierr); /* Replace with proper dgnet creation function */
    dgnet = dgnet_nest->dgnets[i];
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
    /* Command Line Options */
    ierr = PetscOptionsBegin(comm,NULL,"DGNetwork solver options","");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",dgnet->initial,&dgnet->initial,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",dgnet->networktype,&dgnet->networktype,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",dgnet->cfl,&dgnet->cfl,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-length","Length of Edges in the Network","",dgnet->length,&dgnet->length,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",dgnet->Mx,&dgnet->Mx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ndaughters","Number of daughter branches for network type 3","",dgnet->ndaughters,&dgnet->ndaughters,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view","View the DG solution","",dgnet->view,&dgnet->view,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-lax","Use lax curve diagnostic for coupling","",dgnet->laxcurve,&dgnet->laxcurve,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-jumptol","Set jump tolerance for lame one-sided limiter","",dgnet->jumptol,&dgnet->jumptol,NULL);CHKERRQ(ierr);
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
    dgnet->linearcoupling =  PETSC_TRUE; /* only test linear coupling version */
  
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
      ierr = DGNetworkMonitorCreate(dgnet,&dgnet_nest->monitors[i]);CHKERRQ(ierr);
      ierr = DGNetworkAddMonitortoEdges(dgnet,dgnet_nest->monitors[i]);CHKERRQ(ierr);
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
    ierr = RiemannSolverSetUp(dgnet->physics.rs);CHKERRQ(ierr);

    /* Create Vectors */
    ierr = DGNetworkCreateVectors(dgnet);CHKERRQ(ierr);
    /* Set up component dynamic data structures */
    ierr = DGNetworkBuildDynamic(dgnet);CHKERRQ(ierr);
    /* Set up NetRS */
    ierr = DGNetworkAssignNetRS(dgnet,dgnet->physics.rs);CHKERRQ(ierr);
    ierr = DGNetworkProject(dgnet,dgnet->X,0.0);CHKERRQ(ierr);
    Xcomp[i]= dgnet->X;

  }
  /* Create Nest Vectors */
  ierr = VecCreateNest(comm,numsim,NULL,Xcomp,&XNest);CHKERRQ(ierr);
  /* only needsd Xcomp for the creation routine. */ 
  ierr = PetscFree(Xcomp);CHKERRQ(ierr);

  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,dgnet_nest);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(ts,NULL,DGNetRHS_NETRSTEST_Nested,dgnet_nest);CHKERRQ(ierr);

  ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dgnet->cfl/dgnet->Mx/(2*maxorder+1));CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
  if (size == 1 && dgnet->view) {
    ierr = TSMonitorSet(ts, TSDGNetworkMonitor_Nest, dgnet_nest, NULL);CHKERRQ(ierr);
  }
  if (limit) {
      /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */  
    ierr = DGNetlimiter_Nested(ts,0,0,&XNest);CHKERRQ(ierr);
    ierr = TSSetPostStage(ts,DGNetlimiter_Nested);CHKERRQ(ierr);
  } 
  /* Create an extra monitor for viewing the difference */ 
  if (size == 1 && dgnet->view) {
    ierr = DGNetworkMonitorCreate(dgnet,&dgnet_nest->monitors[numsim]);CHKERRQ(ierr);
    ierr = DGNetworkAddMonitortoEdges(dgnet,dgnet_nest->monitors[numsim]);CHKERRQ(ierr);
  }
  /* Create the Vector for holding the difference of solves */ 
  ierr = VecDuplicate(dgnet->X,&Diff);CHKERRQ(ierr); 
  ierr = PetscCalloc1(1,&dgnet_nest->wrk_vec);CHKERRQ(ierr);
  dgnet_nest->wrk_vec[0] = Diff; 
  ierr = TSSetPostStep(ts,TSDGNetworkCompare);CHKERRQ(ierr);
  /* Evolve the PDE network in time */
  ierr = TSSolve(ts,XNest);CHKERRQ(ierr);

 /* Clean up */
  if(dgnet->view && size==1) {
    for (i=0; i<numsim+1; i++) {
      monitor = dgnet_nest->monitors[i]; 
      ierr = DGNetworkMonitorDestroy(&monitor);CHKERRQ(ierr);
    }
  }
  for(i=0; i<numsim; i++) {
    dgnet = dgnet_nest->dgnets[i];
    ierr = RiemannSolverDestroy(&dgnet->physics.rs);CHKERRQ(ierr);
    ierr = PetscFree(dgnet->physics.order);CHKERRQ(ierr);
    ierr = DGNetworkDestroyNetRS(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkDestroy(dgnet);CHKERRQ(ierr); /* Destroy all data within the network and within dgnet */
    ierr = DMDestroy(&dgnet->network);CHKERRQ(ierr);
    ierr = PetscFree(dgnet);CHKERRQ(ierr);
  }
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFree2(dgnet_nest->dgnets,dgnet_nest->monitors);CHKERRQ(ierr); 
  ierr = PetscFree(dgnet_nest->wrk_vec);CHKERRQ(ierr);
  ierr = PetscFree(dgnet_nest);
  
  ierr = VecDestroy(&Diff);CHKERRQ(ierr);
  ierr = VecDestroy(&XNest);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}