static const char help[] = "Traffic Flow Convergence Test for the DGNetwork. \
Uses Method of characteristics to generate a true solution";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"
#include "../physics.h"

/*
  Example : 
  1.  4 levels of convergence with P^4 DG basis and X viewing: 
    mpiexec -np 1 ex4 -convergence 4 -order 4 -view -ts_ssp_type rk104

  2. requires: GLVis 
  4 levels of convergence with P^4 DG basis and GLVis full network view : 
    mpiexec -np 1 ex4 -convergence 4 -order 4 -view -ts_ssp_type rk104 -view_glvis -view_full_net -glvis_pause 1e-10
*/

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
  char              physname[256] = "traffic";
  PetscFunctionList physics = 0;
  MPI_Comm          comm;
  TS                ts;
  DGNetwork         dgnet;
  PetscInt          convergencelevel= 3,maxorder=1,*order,n,i,j;
  PetscReal         maxtime,*norm;
  PetscErrorCode    ierr;
  PetscMPIInt       size,rank;
  DGNetworkMonitor  monitor=NULL;
  DGNetworkMonitor_Glvis monitor_gl=NULL; 
  Vec               Xtrue; 
  PetscBool         glvismode=PETSC_FALSE,view3d=PETSC_FALSE,viewglvis=PETSC_FALSE,viewfullnet = PETSC_FALSE;  

  ierr = PetscInitialize(&argc,&argv,0,help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscMalloc1(1,&dgnet);CHKERRQ(ierr);
  ierr = PetscMemzero(dgnet,sizeof(*dgnet));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"traffic"         ,PhysicsCreate_Traffic);CHKERRQ(ierr);

  /* Set default values */
  dgnet->comm           = comm;
  dgnet->cfl            = 0.5;
  dgnet->networktype    = 6;
  maxtime               = 0.5;
  dgnet->Mx             = 10;
  dgnet->initial        = 1;
  dgnet->ymin           = 0;
  dgnet->ymax           = 2.0;
  dgnet->ndaughters     = 2;
  dgnet->length         = 3.0;
  dgnet->view           = PETSC_FALSE;

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",dgnet->cfl,&dgnet->cfl,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-convergence", "Test convergence on meshes 2^3 - 2^n","",convergencelevel,&convergencelevel,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view","View the DG solution","",dgnet->view,&dgnet->view,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_dump","Dump the Glvis view or socket","",glvismode,&glvismode,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_3d","View a 3d version of edge","",view3d,&view3d,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_glvis","View GLVis of Edge","",viewglvis,&viewglvis,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-view_full_net","View GLVis of Entire Network","",viewfullnet,&viewfullnet,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  {
    PetscErrorCode (*r)(DGNetwork);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(dgnet);CHKERRQ(ierr);
  }

  ierr = PetscMalloc1(dgnet->physics.dof*convergencelevel,&norm);CHKERRQ(ierr);
  ierr = PetscMalloc1(dgnet->physics.dof,&order);CHKERRQ(ierr);
  ierr = MakeOrder(dgnet->physics.dof,order,maxorder);CHKERRQ(ierr);
  /*
  TODO : Build a better physics object with destructor (i.e. use PETSC DS? )
  */
  ierr = (*dgnet->physics.destroy)(dgnet->physics.user);CHKERRQ(ierr);
  for (i=0; i<dgnet->physics.dof; i++) {
    ierr = PetscFree(dgnet->physics.fieldname[i]);CHKERRQ(ierr);
  }
  for (n = 0; n<convergencelevel; ++n) {
    dgnet->Mx = PetscPowInt(2,n); 
    {
      PetscErrorCode (*r)(DGNetwork);
      ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
      if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
      /* Create the physics, will set the number of fields and their names */
      ierr = (*r)(dgnet);CHKERRQ(ierr);
    }
  
    dgnet->physics.order = order;
      /* Generate Network Data */
    ierr = DGNetworkCreate(dgnet,dgnet->networktype,dgnet->Mx);CHKERRQ(ierr);
    /* Create DMNetwork */
    ierr = DMNetworkCreate(PETSC_COMM_WORLD,&dgnet->network);CHKERRQ(ierr);
    if (size == 1 && dgnet->view) {
      if (viewglvis) {
        ierr = DGNetworkMonitorCreate_Glvis(dgnet,&monitor_gl);CHKERRQ(ierr);
      } else {
        ierr = DGNetworkMonitorCreate(dgnet,&monitor);CHKERRQ(ierr);
      }
    }
    /* Set Network Data into the DMNetwork (on proc[0]) */
    ierr = DGNetworkSetComponents(dgnet);CHKERRQ(ierr);
    /* Delete unneeded data in dgnet */
    ierr = DGNetworkCleanUp(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkBuildTabulation(dgnet);CHKERRQ(ierr);
    ierr = DMNetworkDistribute(&dgnet->network,0);CHKERRQ(ierr);
      /* Create Vectors */
    ierr = DGNetworkCreateVectors(dgnet);CHKERRQ(ierr);
    /* Set up component dynamic data structures */
    ierr = DGNetworkBuildDynamic(dgnet);CHKERRQ(ierr);

    if (viewglvis) {
      if (viewfullnet) { 
        ierr =  DGNetworkMonitorAdd_Glvis_2D_NET(monitor_gl,"localhost",glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
      } else {
        if(view3d) {
          ierr = DGNetworkAddMonitortoEdges_Glvis_3D(dgnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        } else {
          ierr = DGNetworkAddMonitortoEdges_Glvis(dgnet,monitor_gl,glvismode ? PETSC_VIEWER_GLVIS_DUMP : PETSC_VIEWER_GLVIS_SOCKET);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = DGNetworkAddMonitortoEdges(dgnet,monitor);CHKERRQ(ierr);
    }
  
    /* Set up Riemann Solver (need a proper riemann physics struct with convienance routine to 
       set all the physics parts at once) */
    ierr = RiemannSolverCreate(dgnet->comm,&dgnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetApplicationContext(dgnet->physics.rs,dgnet->physics.user);CHKERRQ(ierr);
    ierr = RiemannSolverSetFromOptions(dgnet->physics.rs);CHKERRQ(ierr);
    ierr = RiemannSolverSetFluxEig(dgnet->physics.rs,dgnet->physics.fluxeig);CHKERRQ(ierr);
    ierr = RiemannSolverSetFlux(dgnet->physics.rs,1,dgnet->physics.dof,dgnet->physics.flux2);CHKERRQ(ierr);
    ierr = RiemannSolverSetUp(dgnet->physics.rs);CHKERRQ(ierr);
    
    ierr = DGNetworkAssignNetRS(dgnet,dgnet->physics.rs,NULL,-1);CHKERRQ(ierr); /* No Error Estimation or adapativity */
  
    /* Create a time-stepping object */
    ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
    ierr = TSSetDM(ts,dgnet->network);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts,dgnet);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,DGNetRHS_NETRSVERSION2,dgnet);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    /* Compute initial conditions and starting time step */
    ierr = DGNetworkProject(dgnet,dgnet->X,0.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dgnet->cfl/PetscPowReal(2.0,n)/(2*maxorder+1));
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
    if (size == 1 && dgnet->view) {
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
    /* Evolve the PDE network in time */
    ierr = TSSolve(ts,dgnet->X);CHKERRQ(ierr);
    /* Compute true solution and compute norm of the difference with computed solution*/
    ierr = VecDuplicate(dgnet->X,&Xtrue);CHKERRQ(ierr);
    ierr = DGNetworkProject(dgnet,Xtrue,maxtime);CHKERRQ(ierr);
    ierr = VecAXPY(Xtrue,-1,dgnet->X);CHKERRQ(ierr);
    ierr = DGNetworkNormL2(dgnet,Xtrue,norm+(dgnet->physics.dof*(n)));CHKERRQ(ierr);
    ierr = VecDestroy(&Xtrue);CHKERRQ(ierr);

        /* Clean up */
    if(dgnet->view && size==1){
      if(viewglvis) {
        ierr = DGNetworkMonitorDestroy_Glvis(&monitor_gl);
      } else {
        ierr = DGNetworkMonitorDestroy(&monitor);
      }
    }
    ierr = DGNetworkDestroyNetRS(dgnet);CHKERRQ(ierr);
    ierr = RiemannSolverDestroy(&dgnet->physics.rs);CHKERRQ(ierr);
    ierr = DGNetworkDestroy(dgnet);CHKERRQ(ierr); /* Destroy all data within the network and within fvnet */
    ierr = DMDestroy(&dgnet->network);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
  }

  for (j=0;j<dgnet->physics.dof;j++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"My Rank: %i \n",rank);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "DGNET:  Convergence Table for Variable %i \n"
            "DGNET: |---h---||---Error---||---Order---| \n",j);CHKERRQ(ierr);
    for(i=0;i<convergencelevel;i++) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
            "DGNET: |  %g  |  %g  |  %g  | \n",1.0/PetscPowReal(2.0,i),norm[dgnet->physics.dof*i+j],
            !i ? NAN : -PetscLog2Real(norm[dgnet->physics.dof*i+j]/norm[dgnet->physics.dof*(i-1)+j]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }

  ierr = PetscFree(norm);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFree(dgnet);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return ierr;
}