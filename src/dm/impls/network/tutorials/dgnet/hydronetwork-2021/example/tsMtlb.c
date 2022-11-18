static const char help[] = "1D periodic Finite Volume solver with stepping.\n";

/*
  Example: mpiexec -n 2 ./tsMtlb -draw 1 -draw_pause -1
 */

#include <petscts.h>
#include "../src/river.h"

static PetscErrorCode SolutionStatsView(DM da,Vec X,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscReal         xmin,xmax;
  PetscScalar       sum,tvsum,tvgsum;
  const PetscScalar *x;
  PetscInt          imin,imax,Mx,i,j,xs,xm,dof;
  Vec               Xloc;
  PetscBool         iascii;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    /* PETSc lacks a function to compute total variation norm (difficult in multiple dimensions), we do it here */
    ierr  = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMDAVecGetArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr  = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
    ierr  = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
    tvsum = 0;
    for (i=xs; i<xs+xm; i++) {
      for (j=0; j<dof; j++) tvsum += PetscAbsScalar(x[i*dof+j] - x[(i-1)*dof+j]);
    }
    ierr = MPI_Allreduce(&tvsum,&tvgsum,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);

    ierr = VecMin(X,&imin,&xmin);CHKERRQ(ierr);
    ierr = VecMax(X,&imax,&xmax);CHKERRQ(ierr);
    ierr = VecSum(X,&sum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Solution range [%8.5f,%8.5f] with extrema at %D and %D, mean %8.5f, ||x||_TV %8.5f\n",(double)xmin,(double)xmax,imin,imax,(double)(sum/Mx),(double)(tvgsum/Mx));CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,1,"Viewer type not supported");
  PetscFunctionReturn(0);
}

/*
  Input: ts, t (current time), X (current X), Xdot, ctx (river context)
  Ouptut: F = Xdot - f(t,X)
 */
PetscErrorCode RiverIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void* ctx)
{
  PetscErrorCode    ierr;
  River             river=(River)ctx;
  const PetscScalar *xarr,*xdotarr,*xoldarr;
  PetscScalar       *farr;
  DM                da=river->da;
  Vec               Xold;
  RiverField        *f,*x,*xold;
  PetscInt          ncells=river->ncells;
  PetscReal         Uus=0,Hds=1.0;
  
  PetscFunctionBegin;
  ierr = TSGetSolution(ts,&Xold);CHKERRQ(ierr); /* Note: we use Xold, thus an explicit scheme! */

  ierr = DMDAVecGetArrayRead(da,Xold,&xoldarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,X,&xarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdotarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&farr);CHKERRQ(ierr);

  /* Evaluate a single river channel */
  xold = (RiverField*)xoldarr;
  x    = (RiverField*)xarr;
  f    = (RiverField*)farr;
  ierr = RiverIFunctionLocal(river,xold,x,(RiverField*)xdotarr,f);CHKERRQ(ierr);

  /* Set boundaries */
  /* Upstream BC is Q */
  f[0].q = x[0].q - Uus;               
  
  /* Downstream BC is H */
  f[ncells-1].h = x[ncells-1].h - Hds;
  
  ierr = DMDAVecRestoreArrayRead(da,Xold,&xoldarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,X,&xarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdotarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&farr);CHKERRQ(ierr);
  //printf("\n t=%g, F:\n",t);
  //ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* compute initial condition */
static PetscErrorCode RiverSetInitialSolution_tsMtlb(River river,Vec X)
{
  PetscErrorCode ierr;
  PetscInt       i,xs,xm,Mx;
  DM             da=river->da;
  RiverField     *x;
  
  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, 0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++){
    if (i<Mx/2){
      x[i].h = 3.0;
    } else {
      x[i].h = 1.0;
    }
    x[i].q = 0.0;
  }  
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RiverSetUp_tsMtlb(River river)
{ 
  PetscErrorCode ierr;
  PetscInt       s=2;
  PetscReal      dx,length=river->length;

  PetscFunctionBegin;
  /* Create a DMDA to manage the parallel grid */
  ierr = DMDACreate1d(river->comm,DM_BOUNDARY_GHOSTED,river->ncells,2,s,NULL,&river->da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(river->da);CHKERRQ(ierr);
  ierr = DMSetUp(river->da);CHKERRQ(ierr);
 
  ierr = DMDASetFieldName(river->da, 0, "h");CHKERRQ(ierr);
  ierr = DMDASetFieldName(river->da, 1, "Q");CHKERRQ(ierr);
  
  /* Set coordinates of cell centers */
  dx = length/river->ncells;
  ierr = DMDASetUniformCoordinates(river->da,0.5*dx,length+0.5*dx,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  MPI_Comm          comm; 
  DM                da;
  PetscInt          Mx=10;
  PetscReal	    length=5.0; /* spatial step  */
  Vec               X;
  PetscErrorCode    ierr;
  PetscInt          draw = 0,t=0,ntmaxi=3,id=0;//ntmaxi=3
  PetscReal	    gravity=9.8,dt;
  TS                ts;
  River             river;
  Mat               J;
  PetscScalar       *xarr;
  
  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_steps","Num of time steps","",t,&ntmaxi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create river */
  ierr = PetscCalloc1(1,&river);CHKERRQ(ierr);
  river->comm   = comm;
  river->length = length;
  river->ncells = Mx;

  /* SetUp river */
  ierr = RiverSetParameters(river,id,gravity);CHKERRQ(ierr);
  ierr = RiverSetUp_tsMtlb(river);CHKERRQ(ierr);
 
  da = river->da;
 
  /*Create vector */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);

  /*set initial condition. */
  ierr = RiverSetInitialSolution_tsMtlb(river,X);CHKERRQ(ierr);
  if (draw & 0x1) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  //VecView(X,0);

  /* Setup solver                                           */
  /*--------------------------------------------------------*/
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,RiverIFunction,river);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,TSComputeIJacobianDefaultColor,NULL);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,ntmaxi);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetUp(ts);CHKERRQ(ierr);
  //==================================
  
  /*Start time marching procedure */
  for (t=1;t <= ntmaxi; t++){
     printf("\n t=%d  ",t);
     ierr = VecGetArray(X,&xarr);CHKERRQ(ierr); 
     ierr = RiverGetTimeStep(river,(RiverField*)xarr,&dt);CHKERRQ(ierr);
     ierr = VecRestoreArray(X,&xarr);CHKERRQ(ierr); 
  
     ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
     ierr = TSStep(ts);CHKERRQ(ierr);
     ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  
  /* View X */
  ierr = SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  /* Clean up */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&river->da);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree(river);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

