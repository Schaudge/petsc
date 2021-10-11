static const char help[] = "quick riemann solver test. To be deleted at a later date ";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include <petsc/private/kernels/blockinvert.h>
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
PetscErrorCode TSDGNetworkMonitor_GLVis(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor_Glvis   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  ierr = DGNetworkMonitorView_Glvis(monitor,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode TSDGNetworkMonitor_GLVis_NET(TS ts, PetscInt step, PetscReal t, Vec x, void *context)
{
  PetscErrorCode     ierr;
  DGNetworkMonitor_Glvis   monitor;

  PetscFunctionBegin;
  monitor = (DGNetworkMonitor_Glvis)context;
  ierr = DGNetworkMonitorView_Glvis_NET(monitor,x);CHKERRQ(ierr);
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
  char              lname[256] = "minmod",physname[256] = "shallow",tname[256] = "adaptive";
  PetscFunctionList limiters = 0,physics = 0,timestep = 0;
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         fvnet;
  PetscInt          draw = 0,maxorder=1,*order;
  PetscBool         viewdm = PETSC_FALSE,useriemannsolver = PETSC_FALSE;
  PetscReal         maxtime;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  PetscBool         singlecoupleeval,view3d=PETSC_FALSE,viewglvis=PETSC_FALSE,glvismode=PETSC_FALSE,viewfullnet=PETSC_FALSE,limit=PETSC_TRUE;
  DGNetworkMonitor  monitor=NULL;
  DGNetworkMonitor_Glvis monitor_gl = NULL;

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc1(1,&fvnet);CHKERRQ(ierr);
  ierr = PetscMemzero(fvnet,sizeof(*fvnet));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"shallow"         ,PhysicsCreate_Shallow);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&physics,"traffic"         ,PhysicsCreate_Traffic);CHKERRQ(ierr);

  /* Set default values */
  fvnet->comm           = comm;
  fvnet->cfl            = 0.9;
  fvnet->networktype    = 6;
  fvnet->hratio         = 1;
  maxtime               = 2.0;
  fvnet->Mx             = 10;
  fvnet->bufferwidth    = 0;
  fvnet->initial        = 1;
  fvnet->ymin           = 0;
  fvnet->ymax           = 2.0;
  fvnet->bufferwidth    = 4;
  fvnet->ndaughters     = 2;
  fvnet->linearcoupling = PETSC_FALSE;
  singlecoupleeval      = PETSC_FALSE; 
  fvnet->length         = 3.0;
  fvnet->view           = PETSC_TRUE;
  fvnet->jumptol        = 0.5; 

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux imiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-stepsize","Name of function to adapt the timestep size","",timestep,tname,tname,sizeof(tname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-physics","Name of physics model to use","",physics,physname,physname,sizeof(physname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-initial","Initial Condition (depends on the physics)","",fvnet->initial,&fvnet->initial,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-network","Network topology to load, along with boundary condition information","",fvnet->networktype,&fvnet->networktype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact","Compare errors with exact solution","",fvnet->exact,&fvnet->exact,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simulation","Compare errors with reference solution","",fvnet->simulation,&fvnet->simulation,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-lincouple","Use linearized coupling condition when available","",fvnet->linearcoupling,&fvnet->linearcoupling,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",fvnet->cfl,&fvnet->cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-hratio","Spacing ratio","",fvnet->hratio,&fvnet->hratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ts_max_time","Max Time to Run TS","",maxtime,&maxtime,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymin","Min y-value in plotting","",fvnet->ymin,&fvnet->ymin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ymax","Max y-value in plotting","",fvnet->ymax,&fvnet->ymax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-length","Length of Edges in the Network","",fvnet->length,&fvnet->length,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",fvnet->Mx,&fvnet->Mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-bufferwidth","width of the buffer regions","",fvnet->bufferwidth,&fvnet->bufferwidth,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-viewdm","View DMNetwork Info in stdout","",viewdm,&viewdm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ndaughters","Number of daughter branches for network type 3","",fvnet->ndaughters,&fvnet->ndaughters,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-lincouplediff","Compare the results for linearcoupling and nonlinear","",fvnet->lincouplediff,&fvnet->lincouplediff,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-singlecoupleeval","Use the single couple eval rhs functions","",singlecoupleeval,&singlecoupleeval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view","View the DG solution","",fvnet->view,&fvnet->view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_dump","Dump the Glvis view or socket","",glvismode,&glvismode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_3d","View a 3d version of edge","",view3d,&view3d,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_glvis","View GLVis of Edge","",viewglvis,&viewglvis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_full_net","View GLVis of Entire Network","",viewfullnet,&viewfullnet,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-uselimiter","Use a limiter for the DG solution","",limit,&limit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-jumptol","Set jump tolerance for lame one-sided limiter","",fvnet->jumptol,&fvnet->jumptol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-useriemannsolver","use the riemann solver class","",useriemannsolver,&useriemannsolver,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  fvnet->linearcoupling = singlecoupleeval;
  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(DGNetwork);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(fvnet);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(fvnet->physics.dof,&order);CHKERRQ(ierr);
  ierr = MakeOrder(fvnet->physics.dof,order,maxorder);CHKERRQ(ierr);
  fvnet->physics.order = order;
  /* Generate Network Data */
  ierr = DGNetworkCreate(fvnet,fvnet->networktype,fvnet->Mx);CHKERRQ(ierr);
  /* Create DMNetwork */
  ierr = DMNetworkCreate(PETSC_COMM_WORLD,&fvnet->network);CHKERRQ(ierr);

  if (size == 1 && fvnet->view) {
    if (viewglvis) {
      ierr = DGNetworkMonitorCreate_Glvis(fvnet,&monitor_gl);CHKERRQ(ierr);
    } else {
      ierr = DGNetworkMonitorCreate(fvnet,&monitor);CHKERRQ(ierr);
    }
  }
  /* Set Network Data into the DMNetwork (on proc[0]) */
  ierr = DGNetworkSetComponents(fvnet);CHKERRQ(ierr);
  /* Delete unneeded data in fvnet */
  ierr = DGNetworkCleanUp(fvnet);CHKERRQ(ierr);
  ierr = DGNetworkBuildTabulation(fvnet);CHKERRQ(ierr);
    if (viewglvis) {
      if (viewfullnet) { 
        ierr =  DGNetworkMonitorAdd_Glvis_3D_NET(monitor_gl,"localhost",glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
      } else {
        if(view3d) {
          ierr = DGNetworkAddMonitortoEdges_Glvis_3D(fvnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        } else {
          ierr = DGNetworkAddMonitortoEdges_Glvis(fvnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = DGNetworkAddMonitortoEdges(fvnet,monitor);CHKERRQ(ierr);
    }

  /* Set up Riemann Solver */
  if(useriemannsolver) 
  {
    ierr = RiemannSolverCreate(fvnet->comm,&fvnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(fvnet->physics.rs,fvnet->physics.user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(fvnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(fvnet->physics.rs,fvnet->physics.fluxeig);CHKERRQ(ierr);
    ierr = RiemannSolverSetRoeAvgFunct(fvnet->physics.rs,fvnet->physics.roeavg);CHKERRQ(ierr);
    ierr = RiemannSolverSetEigBasis(fvnet->physics.rs,fvnet->physics.eigbasis);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(fvnet->physics.rs,1,fvnet->physics.dof,fvnet->physics.flux2);CHKERRQ(ierr); 
    ierr = RiemannSolverSetUp(fvnet->physics.rs);CHKERRQ(ierr);
  }  

  /* Create Vectors */
  ierr = DGNetworkCreateVectors(fvnet);CHKERRQ(ierr);
  /* Set up component dynamic data structures */
  ierr = DGNetworkBuildDynamic(fvnet);CHKERRQ(ierr);
  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,fvnet->network);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts,fvnet);CHKERRQ(ierr);
  if(useriemannsolver)
  {
    ierr = TSSetRHSFunction(ts,NULL,DGNetRHS_RSVERSION,fvnet);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSFunction(ts,NULL,DGNetRHS,fvnet);CHKERRQ(ierr);
  }
  ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,fvnet->cfl/fvnet->Mx/(2*maxorder+1));CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = DGNetworkProject(fvnet,fvnet->X,0.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
   if (size == 1 && fvnet->view) {
      if (viewglvis) {
        if(viewfullnet) {
          ierr = TSMonitorSet(ts, TSDGNetworkMonitor_GLVis_NET, monitor_gl, NULL);CHKERRQ(ierr);
        } else {
          ierr = TSMonitorSet(ts, TSDGNetworkMonitor_GLVis, monitor_gl, NULL);CHKERRQ(ierr);
        } 
      } else {
        ierr = TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL);CHKERRQ(ierr);
      }
    }
  if (limit) {
      /* Prelimit the initial data as I use post-stage to apply limiters instead of prestage (which doesn't have access to stage vectors 
  for some reason ... no idea why prestage and post-stage callback functions have different forms) */  
    ierr = DGNetlimiter(ts,0,0,&fvnet->X);CHKERRQ(ierr);
    ierr = TSSetPostStage(ts,DGNetlimiter);CHKERRQ(ierr);
  } 

  /* Evolve the PDE network in time */
  ierr = TSSolve(ts,fvnet->X);CHKERRQ(ierr);

  /* Clean up */
  if(fvnet->view && size==1){
    if(viewglvis) {
      ierr = DGNetworkMonitorDestroy_Glvis(&monitor_gl);
    } else {
      ierr = DGNetworkMonitorDestroy(&monitor);
    }
  } 
  ierr = RiemannSolverDestroy(&fvnet->physics.rs);CHKERRQ(ierr);
  ierr = DGNetworkDestroy(fvnet);CHKERRQ(ierr); /* Destroy all data within the network and within fvnet */
  ierr = DMDestroy(&fvnet->network);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&timestep);CHKERRQ(ierr);
  ierr = PetscFree(fvnet);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}