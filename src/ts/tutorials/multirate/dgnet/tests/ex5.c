static const char help[] = "Traffic Flow Convergence Test for the DGNetwork";

#include <petscts.h>
#include <petscdm.h>
#include <petscdraw.h>
#include <petscdmnetwork.h>
#include "../dgnet.h"

PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }

/* --------------------------------- Traffic ----------------------------------- */

typedef struct {
  PetscReal a;
} TrafficCtx;

PETSC_STATIC_INLINE PetscScalar TrafficFlux(PetscScalar a,PetscScalar u) { return a*u*(1-u); }
PETSC_STATIC_INLINE PetscErrorCode TrafficFlux2(void *ctx,const PetscReal *u,PetscReal *f) {
  TrafficCtx *phys = (TrafficCtx*)ctx;
  f[0] = phys->a * u[0]*(1. - u[0]);
  PetscFunctionReturn(0);
}
PETSC_STATIC_INLINE PetscScalar TrafficChar(PetscScalar a,PetscScalar u) { return a*(1-2*u); }

static PetscErrorCode PhysicsRiemann_Traffic_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;

  PetscFunctionBeginUser;
  if (uL[0] < uR[0]) {
    flux[0] = PetscMin(TrafficFlux(a,uL[0]),TrafficFlux(a,uR[0]));
  } else {
    flux[0] = (uR[0] < 0.5 && 0.5 < uL[0]) ? TrafficFlux(a,0.5) : PetscMax(TrafficFlux(a,uL[0]),TrafficFlux(a,uR[0]));
  }
  *maxspeed = a*MaxAbs(1-2*uL[0],1-2*uR[0]);
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Traffic_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed = a*(1 - (uL[0] + uR[0]));
  flux[0] = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*PetscAbs(speed)*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsRiemann_Traffic_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBeginUser;
  speed     = a*PetscMax(PetscAbs(1-2*uL[0]),PetscAbs(1-2*uR[0]));
  flux[0]   = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*speed*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}
/*2 edge vertex flux for edge 1 pointing in and edge 2 pointing out */
static PetscErrorCode PhysicsVertexFlux_2Edge_InOut(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uL, 2nd as uR. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV,uV+dof,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

/*2 edge vertex flux for edge 1 pointing out and edge 2 pointing in  */
static PetscErrorCode PhysicsVertexFlux_2Edge_OutIn(const void* _dgnet,const PetscScalar *uV,const PetscBool *dir,PetscScalar *flux,PetscScalar *maxspeed,const void* _junct)
{
  PetscErrorCode  ierr;
  const DGNetwork fvnet = (DGNetwork)_dgnet;
  PetscInt        i,dof = fvnet->physics.dof;

  PetscFunctionBeginUser;
  /* First edge interpreted as uR, 2nd as uL. Use the user inputted Riemann function. */
  ierr = fvnet->physics.riemann(fvnet->physics.user,dof,uV+dof,uV,flux,maxspeed);CHKERRQ(ierr);
  /* Copy the flux */
  for (i = 0; i<dof; i++) {
    flux[i+dof] = flux[i];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsAssignVertexFlux_Traffic(const void* _fvnet, Junction junct)
{  
  PetscFunctionBeginUser;
      if (junct->numedges == 2) {
        if (junct->dir[0] == EDGEIN) {
          if (junct->dir[1] == EDGEIN) {
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          } else { /* dir[1] == EDGEOUT */
            junct->couplingflux = PhysicsVertexFlux_2Edge_InOut;
          }
        } else { /* dir[0] == EDGEOUT */
          if (junct->dir[1] == EDGEIN) {
            junct->couplingflux = PhysicsVertexFlux_2Edge_OutIn;
          } else { /* dir[1] == EDGEOUT */
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not a valid directed graph for the current discretization method");
          }
        }
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"General coupling conditions are not yet implemented for traffic models");
      }
  PetscFunctionReturn(0);
}
typedef struct {
  PetscReal a,x,t;
} MethodCharCtx; 
/* TODO Generalize to arbitrary initial value */
static  PetscErrorCode TrafficCase1Char(SNES snes,Vec X,Vec f, void *ctx) {
  PetscReal      x,t,rhs,a;
  const PetscScalar *s; 
  PetscErrorCode ierr; 
  
  PetscFunctionBeginUser;
  x = ((MethodCharCtx*)ctx)->x;
  t = ((MethodCharCtx*)ctx)->t;
  a = ((MethodCharCtx*)ctx)->a;

  ierr = VecGetArrayRead(X,&s);CHKERRQ(ierr);
  rhs  = TrafficChar(a,PetscSinReal(PETSC_PI*(s[0]/5.0))+2)*t +s[0]-x; 
  ierr = VecSetValue(f,0,rhs,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* TODO Generalize to arbitrary initial value */
static  PetscErrorCode TrafficCase1Char_J(SNES snes,Vec X,Mat Amat,Mat Pmat, void *ctx) {
  PetscReal      x,t,rhs,a;
  const PetscScalar *s; 
  PetscErrorCode ierr; 
  
  PetscFunctionBeginUser;
  x = ((MethodCharCtx*)ctx)->x;
  t = ((MethodCharCtx*)ctx)->t;
  a = ((MethodCharCtx*)ctx)->a;

  ierr = VecGetArrayRead(X,&s);CHKERRQ(ierr);
  rhs = 1.0- t*a*2.0*PETSC_PI/5.0*PetscCosReal(PETSC_PI*(s[0]/5.0)); 
  ierr = MatSetValue(Pmat,0,0,rhs,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&s);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) {
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsSample_TrafficNetwork(void *vctx,PetscInt initial,PetscReal t,PetscReal x,PetscReal *u,PetscInt edgeid)
{
  SNES           snes; 
  Mat            J;
  Vec            X,R;
  PetscErrorCode ierr; 
  PetscReal      *s;
  MethodCharCtx  ctx;  

  PetscFunctionBeginUser;
  if (t<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"t must be >= 0 ");
  switch (initial) {
    case 0:
        if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exact solution for case 0 not implemented for t > 0");
      if(edgeid == 0) {
        u[0] = (x < -2) ? 2 : 1; /* Traffic Break problem ?*/
      } else {
        u[0] = 1; 
      }
      break;
    case 1:
      if (t==0.0) {
        if(edgeid == 0) {
          u[0] = PetscSinReal(PETSC_PI*(x/5.0))+2;
        } else if(edgeid == 1) {
          u[0] = PetscSinReal(PETSC_PI*(-x/5.0))+2;
        } else {
          u[0] = 0; 
        }
      } else {
          /* Method of characteristics to solve for exact solution */
          ctx.t =t; ctx.a = 0.5;
          ctx.x = !edgeid ? x : -x; 
          ierr = VecCreate(PETSC_COMM_SELF,&X);CHKERRQ(ierr);
          ierr = VecSetSizes(X,PETSC_DECIDE,1);CHKERRQ(ierr);
          ierr = VecSetFromOptions(X);CHKERRQ(ierr);
          ierr = VecDuplicate(X,&R);CHKERRQ(ierr);
          ierr = MatCreate(PETSC_COMM_SELF,&J);CHKERRQ(ierr);
          ierr = MatSetSizes(J,1,1,1,1);CHKERRQ(ierr);
          ierr = MatSetFromOptions(J);CHKERRQ(ierr);
          ierr = MatSetUp(J);CHKERRQ(ierr);
          ierr = SNESCreate(PETSC_COMM_SELF,&snes);CHKERRQ(ierr);
          ierr = SNESSetFunction(snes,R,TrafficCase1Char,&ctx);
          ierr = SNESSetJacobian(snes,J,J,TrafficCase1Char_J,&ctx);
          ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
          ierr = VecSet(X,x);CHKERRQ(ierr);
          ierr = SNESSolve(snes,NULL,X);CHKERRQ(ierr);
          ierr = VecGetArray(X,&s);CHKERRQ(ierr);
      
          u[0] = PetscSinReal(PETSC_PI*(s[0]/5.0))+2;
         
          ierr = VecRestoreArray(X,&s);CHKERRQ(ierr);
          ierr = VecDestroy(&X);CHKERRQ(ierr);
          ierr = VecDestroy(&R);CHKERRQ(ierr);
          ierr = MatDestroy(&J);CHKERRQ(ierr);
          ierr = SNESDestroy(&snes);CHKERRQ(ierr);
      } 
      break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsDestroyVertexFlux(const void* _fvnet, Junction junct)
{
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
          ierr = VecDestroy(&junct->rcouple);CHKERRQ(ierr);
          ierr = VecDestroy(&junct->xcouple);CHKERRQ(ierr);
          ierr = MatDestroy(&junct->mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds)
{
  PetscInt i,j;

  PetscFunctionBeginUser;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    speeds[i] = PETSC_MAX_REAL; /* Indicates invalid */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Traffic(DGNetwork fvnet)
{
  PetscErrorCode    ierr;
  TrafficCtx        *user;
  RiemannFunction   r;
  PetscFunctionList rlist      = 0;
  char              rname[256] = "rusanov";

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  fvnet->physics.samplenetwork  = PhysicsSample_TrafficNetwork;
  fvnet->physics.characteristic = PhysicsCharacteristic_Conservative;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.user           = user;
  fvnet->physics.dof            = 1;
  fvnet->physics.destroy        = PhysicsDestroy_SimpleFree_Net;
  fvnet->physics.vfluxassign    = PhysicsAssignVertexFlux_Traffic;
  fvnet->physics.vfluxdestroy   = PhysicsDestroyVertexFlux;
  fvnet->physics.flux           = TrafficFlux2;
  fvnet->physics.user           = user;

  ierr = PetscStrallocpy("density",&fvnet->physics.fieldname[0]);CHKERRQ(ierr);
  user->a = 0.5;
  ierr = RiemannListAdd_Net(&rlist,"rusanov",PhysicsRiemann_Traffic_Rusanov);CHKERRQ(ierr);
  ierr = RiemannListAdd_Net(&rlist,"exact",  PhysicsRiemann_Traffic_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd_Net(&rlist,"roe",    PhysicsRiemann_Traffic_Roe);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(fvnet->comm,fvnet->prefix,"Options for Traffic","");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_traffic_a","Flux = a*u*(1-u)","",user->a,&user->a,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsFList("-physics_traffic_riemann","Riemann solver","",rlist,rname,rname,sizeof(rname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = RiemannListFind_Net(rlist,rname,&r);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&rlist);CHKERRQ(ierr);

  fvnet->physics.riemann = r;
  PetscFunctionReturn(0);
}

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
  Vec               Xtrue; 

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
  dgnet->hratio         = 2;
  maxtime               = 0.5;
  dgnet->Mx             = 10;
  dgnet->bufferwidth    = 0;
  dgnet->initial        = 1;
  dgnet->ymin           = 0;
  dgnet->ymax           = 2.0;
  dgnet->bufferwidth    = 4;
  dgnet->ndaughters     = 2;
  dgnet->linearcoupling = PETSC_FALSE;
  dgnet->length         = 3.0;
  dgnet->view           = PETSC_TRUE;

  /* Command Line Options */
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",dgnet->cfl,&dgnet->cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Mx","Smallest number of cells for an edge","",dgnet->Mx,&dgnet->Mx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-convergence", "Test convergence on meshes 2^3 - 2^n","",convergencelevel,&convergencelevel,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-order", "Order of the DG Basis","",maxorder,&maxorder,NULL);CHKERRQ(ierr);
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
      ierr = DGNetworkMonitorCreate(dgnet,&monitor);CHKERRQ(ierr);
    }
    /* Set Network Data into the DMNetwork (on proc[0]) */
    ierr = DGNetworkSetComponents(dgnet);CHKERRQ(ierr);
    /* Delete unneeded data in dgnet */
    ierr = DGNetworkCleanUp(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkBuildTabulation(dgnet);CHKERRQ(ierr);
    ierr = DGNetworkAddMonitortoEdges(dgnet,monitor);CHKERRQ(ierr);
    /* Create Vectors */
    ierr = DGNetworkCreateVectors(dgnet);CHKERRQ(ierr);
    /* Set up component dynamic data structures */
    ierr = DGNetworkBuildDynamic(dgnet);CHKERRQ(ierr);
    /* Create a time-stepping object */
    ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
    ierr = TSSetDM(ts,dgnet->network);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts,dgnet);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,DGNetRHS,dgnet);CHKERRQ(ierr);
    ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,maxtime);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

    /* Compute initial conditions and starting time step */
    ierr = DGNetworkProject(dgnet,dgnet->X,0.0);CHKERRQ(ierr);
    ierr = DGNetRHS(ts,0,dgnet->X,dgnet->Ftmp,dgnet);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dgnet->cfl/PetscPowReal(2.0,n)/(2*maxorder+1));
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);  /* Take runtime options */
    if (size == 1 && dgnet->view) ierr = TSMonitorSet(ts, TSDGNetworkMonitor, monitor, NULL);CHKERRQ(ierr);
    /* Evolve the PDE network in time */
    ierr = TSSolve(ts,dgnet->X);CHKERRQ(ierr);
    /* Compute true solution and compute norm of the difference with computed solution*/
    ierr = VecDuplicate(dgnet->X,&Xtrue);CHKERRQ(ierr);
    ierr = DGNetworkProject(dgnet,Xtrue,maxtime);CHKERRQ(ierr);
    ierr = VecAXPY(Xtrue,-1,dgnet->X);CHKERRQ(ierr);
    ierr = DGNetworkNormL2(dgnet,Xtrue,norm+(dgnet->physics.dof*(n)));CHKERRQ(ierr);
    ierr = VecDestroy(&Xtrue);CHKERRQ(ierr);

        /* Clean up */
    if(dgnet->view && size==1) ierr = DGNetworkMonitorDestroy(&monitor);
    ierr = DGNetworkDestroy(dgnet);CHKERRQ(ierr); /* Destroy all data within the network and within fvnet */
    ierr = DMDestroy(&dgnet->network);CHKERRQ(ierr);
    ierr = TSDestroy(&ts);CHKERRQ(ierr);
  }

  for (j=0;j<dgnet->physics.dof;j++) {
    ierr = PetscPrintf(comm,
            "DGNET:  Convergence Table for Variable %i \n"
            "DGNET: |---h---||---Error---||---Order---| \n",j);CHKERRQ(ierr);
    for(i=0;i<convergencelevel;i++) {
      ierr = PetscPrintf(comm,
            "DGNET: |  %g  |  %g  |  %g  | \n",1.0/PetscPowReal(2.0,i),norm[dgnet->physics.dof*i+j],
            !i ? NAN : -PetscLog2Real(norm[dgnet->physics.dof*i+j]/norm[dgnet->physics.dof*(i-1)+j]));CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
  }

  ierr = PetscFree(norm);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFree(dgnet);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}