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
  DGNetworkMonitor   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor)context;
  PetscCall(DGNetworkMonitorView(monitor,x));
  PetscFunctionReturn(0);
}
PetscErrorCode TSDGNetworkMonitor_Nest(TS ts, PetscInt step, PetscReal t, Vec x, void *ctx)
{
  DGNetwork_Nest     dgnet_nest = (DGNetwork_Nest)ctx; 
  PetscInt           i;
  Vec                Xsim; 
  DGNetwork          dgnet;

  PetscFunctionBegin;
  for(i=0; i<dgnet_nest->numsimulations; i++) {
    PetscCall(VecNestGetSubVec(x,i,&Xsim));
    dgnet = dgnet_nest->dgnets[i];
    if(dgnet->viewglvis) {
      if(dgnet->viewfullnet) {
        PetscCall(DGNetworkMonitorView_Glvis_NET(dgnet_nest->monitors_glvis[i],Xsim));
      } else {
        PetscCall(DGNetworkMonitorView_Glvis(dgnet_nest->monitors_glvis[i],Xsim));
      }
    } else {
      PetscCall(DGNetworkMonitorView(dgnet_nest->monitors[i],Xsim));
    }
  }
  PetscFunctionReturn(0);
}
/* poststep function for comparing the two coupling conditions */
PetscErrorCode TSDGNetworkCompare(TS ts)
{
  DGNetwork_Nest     dgnet_nest;
  Vec                Xsim,Xnest,Xdiff; 
  MPI_Comm           comm; 
  DGNetwork          dgnet; 

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts,&dgnet_nest));
  PetscCall(PetscObjectGetComm((PetscObject)ts,&comm));
  Xdiff = dgnet_nest->wrk_vec[0];
  PetscCall(TSGetSolution(ts,&Xnest));
  if(dgnet_nest->numsimulations !=2) {SETERRQ1(comm,PETSC_ERR_SUP,"Only valid for 2 simulations. \n \
  There were %i simulations",dgnet_nest->numsimulations);}
  PetscCall(VecNestGetSubVec(Xnest,0,&Xsim));
  PetscCall(VecCopy(Xsim,Xdiff)); 
  PetscCall(VecNestGetSubVec(Xnest,1,&Xsim));
  PetscCall(VecAXPY(Xdiff,-1.,Xsim));
  dgnet = dgnet_nest->dgnets[0];
  if(dgnet_nest->monitors_glvis[2] || dgnet_nest->monitors[2]) {
    if(dgnet->viewglvis) {
      if(dgnet->viewfullnet) {
        PetscCall(DGNetworkMonitorView_Glvis_NET(dgnet_nest->monitors_glvis[2],Xdiff));
      } else {
        PetscCall(DGNetworkMonitorView_Glvis(dgnet_nest->monitors_glvis[2],Xdiff));
      }
    } else {
      PetscCall(DGNetworkMonitorView(dgnet_nest->monitors[2],Xdiff));
    }  
  }
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
  char              physname[256] = "shallow",errorestimator[256] = "lax",filename[256]="localhost";
  PetscFunctionList physics = 0,errest = 0; 
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         dgnet;
  PetscInt          i,maxorder=1,numsim=2;
  PetscReal         maxtime,adapttol=1,adapttol2=10;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscBool         limit=PETSC_TRUE,view3d=PETSC_FALSE,viewglvis=PETSC_FALSE,glvismode=PETSC_FALSE,viewfullnet=PETSC_FALSE;
  NRSErrorEstimator errorest; 
  DGNetwork_Nest    dgnet_nest;
  Vec               Diff, XNest,*Xcomp; 

  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow));

    /* register error estimator functions */
    PetscCall(PetscFunctionListAdd(&errest,"roe"         ,NetRSRoeErrorEstimate));
    PetscCall(PetscFunctionListAdd(&errest,"lax"         ,NetRSLaxErrorEstimate));
    PetscCall(PetscFunctionListAdd(&errest,"taylor"      ,NetRSTaylorErrorEstimate));

  /* Build the nest structure */ 
  PetscCall(PetscCalloc1(1,&dgnet_nest));
  dgnet_nest->numsimulations = numsim;
  PetscCall(PetscCalloc3(numsim,&dgnet_nest->dgnets,numsim+1,&dgnet_nest->monitors,numsim+1,&dgnet_nest->monitors_glvis));
  PetscCall(PetscCalloc1(numsim,&Xcomp));
  /* loop the DGNetwork setup
    Note: This is way too long of code for setting things up, need to break things down more 
  */
 /* clean up the output directory */
  PetscCall(PetscRMTree("output"));
  PetscCall(PetscMkdir("output"));
  for(i=0; i<numsim; i++) {
    PetscCall(PetscCalloc1(1,&dgnet_nest->dgnets[i])); /* Replace with proper dgnet creation function */
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
    PetscCall(PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL));
    PetscCall(PetscOptionsFList("-errest","","",errest,errorestimator,errorestimator,sizeof(physname),NULL));
    PetscCall(PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",dgnet->initial,&dgnet->initial,NULL));
    PetscCall(PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",dgnet->networktype,&dgnet->networktype,NULL));
    PetscCall(PetscOptionsReal("-cfl","CFL number to time step at","",dgnet->cfl,&dgnet->cfl,NULL));
    PetscCall(PetscOptionsReal("-length","Length of Edges in the Network","",dgnet->length,&dgnet->length,NULL));
    PetscCall(PetscOptionsReal("-adapttol","","",adapttol,&adapttol,NULL));
    PetscCall(PetscOptionsReal("-adapttol2","","",adapttol2,&adapttol2,NULL));
    PetscCall(PetscOptionsInt("-Mx","Smallest number of cells for an edge","",dgnet->Mx,&dgnet->Mx,NULL));
    PetscCall(PetscOptionsInt("-ndaughters","Number of daughter branches for network type 3","",dgnet->ndaughters,&dgnet->ndaughters,NULL));
    PetscCall(PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL));
    PetscCall(PetscOptionsBool("-view","View the DG solution","",dgnet->view,&dgnet->view,NULL));
    PetscCall(PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL));
    PetscCall(PetscOptionsBool("-lax","Use lax curve diagnostic for coupling","",dgnet->laxcurve,&dgnet->laxcurve,NULL));
    PetscCall(PetscOptionsReal("-jumptol","Set jump tolerance for lame one-sided limiter","",dgnet->jumptol,&dgnet->jumptol,NULL));
    PetscCall(PetscOptionsBool("-lincouple","Use lax curve diagnostic for coupling","",dgnet->linearcoupling,&dgnet->linearcoupling,NULL));
    PetscCall(PetscOptionsBool("-view_dump","Dump the Glvis view or socket","",glvismode,&glvismode,NULL));
    PetscCall(PetscOptionsBool("-view_3d","View a 3d version of edge","",view3d,&view3d,NULL));
    PetscCall(PetscOptionsBool("-view_glvis","View GLVis of Edge","",viewglvis,&viewglvis,NULL));
    PetscCall(PetscOptionsBool("-view_full_net","View GLVis of Entire Network","",viewfullnet,&viewfullnet,NULL));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    /* Choose the physics from the list of registered models */
    {
      PetscErrorCode (*r)(DGNetwork);
      PetscCall(PetscFunctionListFind(physics,physname,&r));
      if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
      /* Create the physics, will set the number of fields and their names */
      PetscCall((*r)(dgnet));
    }
    PetscCall(PetscMalloc1(dgnet->physics.dof,&dgnet->physics.order)); /* should be constructed by physics */
    PetscCall(MakeOrder(dgnet->physics.dof,dgnet->physics.order,maxorder));
    dgnet->linearcoupling =  PETSC_TRUE; /* only test linear coupling version */
  
    /* Generate Network Data */
    PetscCall(DGNetworkCreate(dgnet,dgnet->networktype,dgnet->Mx));
    /* Create DMNetwork */
    PetscCall(DMNetworkCreate(PETSC_COMM_WORLD,&dgnet->network));
    
    /* Set Network Data into the DMNetwork (on proc[0]) */
    PetscCall(DGNetworkSetComponents(dgnet));
    /* Delete unneeded data in dgnet */
    PetscCall(DGNetworkCleanUp(dgnet));
    PetscCall(DGNetworkBuildTabulation(dgnet));
    PetscCall(DMNetworkDistribute(&dgnet->network,0));
    /* Create Vectors */
    PetscCall(DGNetworkCreateVectors(dgnet));
    /* Set up component dynamic data structures */
    PetscCall(DGNetworkBuildDynamic(dgnet));


  if(glvismode){
    PetscCall(PetscSNPrintf(filename,256,"./output/Sim%i",i));
       PetscCall(PetscMkdir(filename));
           PetscCall(PetscSNPrintf(filename,256,"./output/Sim%i/Hi",i));

  }
  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorCreate_Glvis(dgnet,&dgnet_nest->monitors_glvis[i]));
      if (viewfullnet) { 
        PetscCall(DGNetworkMonitorAdd_Glvis_2D_NET(dgnet_nest->monitors_glvis[i],filename,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
      } else {
        if(view3d) {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis_3D(dgnet,dgnet_nest->monitors_glvis[i],glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        } else {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis(dgnet,dgnet_nest->monitors_glvis[i],glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        }
      }
    } else {
      PetscCall(DGNetworkMonitorCreate(dgnet,&dgnet_nest->monitors[i]));
      PetscCall(DGNetworkAddMonitortoEdges(dgnet,dgnet_nest->monitors[i]));
    }
  }
  
    /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to 
       set all the physics parts at once) */
    PetscCall(RiemannSolverCreate(dgnet->comm,&dgnet->physics.rs));
    PetscCall(RiemannSolverSetApplicationContext(dgnet->physics.rs,dgnet->physics.user));
    PetscCall(RiemannSolverSetFromOptions(dgnet->physics.rs));
    PetscCall(RiemannSolverSetFluxEig(dgnet->physics.rs,dgnet->physics.fluxeig));
    PetscCall(RiemannSolverSetRoeAvgFunct(dgnet->physics.rs,dgnet->physics.roeavg));
    PetscCall(RiemannSolverSetRoeMatrixFunct(dgnet->physics.rs,dgnet->physics.roemat));
    PetscCall(RiemannSolverSetEigBasis(dgnet->physics.rs,dgnet->physics.eigbasis));
    PetscCall(RiemannSolverSetFlux(dgnet->physics.rs,1,dgnet->physics.dof,dgnet->physics.flux2));
    PetscCall(RiemannSolverSetLaxCurve(dgnet->physics.rs,dgnet->physics.laxcurve));
    PetscCall(RiemannSolverSetUp(dgnet->physics.rs));
    
    /* Set up NetRS */
    PetscCall(PetscFunctionListFind(errest,errorestimator,&errorest));
    if (i==0) {
      PetscCall(DGNetworkAssignNetRS(dgnet,dgnet->physics.rs,errorest,adapttol));
    } else {
      PetscCall(DGNetworkAssignNetRS(dgnet,dgnet->physics.rs,errorest,adapttol2));
    }
    PetscCall(DGNetworkProject(dgnet,dgnet->X,0.0));
    Xcomp[i]= dgnet->X;
    dgnet->viewglvis= viewglvis; 
    dgnet->viewfullnet=viewfullnet; 
  }
  /* Create Nest Vectors */
  PetscCall(VecCreateNest(comm,numsim,NULL,Xcomp,&XNest));
  /* only needsd Xcomp for the creation routine. */ 
  PetscCall(PetscFree(Xcomp));

  /* Create a time-stepping object */
  PetscCall(TSCreate(comm,&ts));
  PetscCall(TSSetApplicationContext(ts,dgnet_nest));
  
  PetscCall(TSSetRHSFunction(ts,NULL,DGNetRHS_NETRS_Nested,dgnet_nest));

  PetscCall(TSSetType(ts,TSSSP));
  PetscCall(TSSetMaxTime(ts,maxtime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts,dgnet->cfl/dgnet->Mx/(2*maxorder+1)));

  PetscCall(TSSetFromOptions(ts));  /* Take runtime options */
  if (size == 1 && dgnet->view) {
    PetscCall(TSMonitorSet(ts, TSDGNetworkMonitor_Nest, dgnet_nest, NULL));
  }
  if (limit) {
      /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
      for some reason ... no idea why prestage and post-stage callback functions have different forms) */  
    PetscCall(DGNetlimiter_Nested(ts,0,0,&XNest));
    PetscCall(TSSetPostStage(ts,DGNetlimiter_Nested));
  } 
  /* Create an extra monitor for viewing the difference */ 

  if(glvismode){
    
    PetscCall(PetscSNPrintf(filename,256,"./output/Error"));
    PetscCall(PetscMkdir(filename));
      PetscCall(PetscSNPrintf(filename,256,"./output/Error/Hi"));
  }
  if (size == 1 && dgnet->view) {
    if (viewglvis) {
      PetscCall(DGNetworkMonitorCreate_Glvis(dgnet,&dgnet_nest->monitors_glvis[numsim]));
      if (viewfullnet) { 
        PetscCall(DGNetworkMonitorAdd_Glvis_2D_NET(dgnet_nest->monitors_glvis[numsim],filename,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
      } else {
        if(view3d) {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis_3D(dgnet,dgnet_nest->monitors_glvis[numsim],glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        } else {
          PetscCall(DGNetworkAddMonitortoEdges_Glvis(dgnet,dgnet_nest->monitors_glvis[numsim],glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET));
        }
      }
    } else {
      PetscCall(DGNetworkMonitorCreate(dgnet,&dgnet_nest->monitors[numsim]));
      PetscCall(DGNetworkAddMonitortoEdges(dgnet,dgnet_nest->monitors[numsim]));
    }
  }
  /* Create the Vector for holding the difference of solves */ 
  PetscCall(VecDuplicate(dgnet->X,&Diff)); 
  PetscCall(PetscCalloc1(1,&dgnet_nest->wrk_vec));
  dgnet_nest->wrk_vec[0] = Diff; 
  PetscCall(TSSetPostStep(ts,TSDGNetworkCompare));
  /* Evolve the PDE network in time */
  PetscCall(TSSolve(ts,XNest));

 /* Clean up */
  if(dgnet->view && size==1) {
    for (i=0; i<numsim+1; i++) {
      if(!dgnet->viewglvis) {
        PetscCall(DGNetworkMonitorDestroy(&dgnet_nest->monitors[i]));
      } else {
        PetscCall(DGNetworkMonitorDestroy_Glvis(&dgnet_nest->monitors_glvis[i]));
      }
    }
  }
  for(i=0; i<numsim; i++) {
    dgnet = dgnet_nest->dgnets[i];
    PetscCall(RiemannSolverDestroy(&dgnet->physics.rs));
    PetscCall(PetscFree(dgnet->physics.order));
    PetscCall(DGNetworkDestroyNetRS(dgnet));
    PetscCall(DGNetworkDestroy(dgnet)); /* Destroy all data within the network and within dgnet */
    PetscCall(DMDestroy(&dgnet->network));
    PetscCall(PetscFree(dgnet));
  }
  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFree3(dgnet_nest->dgnets,dgnet_nest->monitors,dgnet_nest->monitors_glvis)); 
  PetscCall(PetscFree(dgnet_nest->wrk_vec));
  ierr = PetscFree(dgnet_nest);
  
  PetscCall(VecDestroy(&Diff));
  PetscCall(VecDestroy(&XNest));

  PetscCall(PetscFinalize();CHKERRQ(ierr));
  return 0;
}
