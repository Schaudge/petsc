/*
  Example 01

  Compile with: make ex-01

  Sample run:   mpirun -np 2 ex-01

  Description:

  Blah...

  When run with the default 10 time steps,the solution is as follows:

     1.00000000000000e+00
     5.00000000000000e-01
     2.50000000000000e-01
     1.25000000000000e-01
     6.25000000000000e-02
     3.12500000000000e-02
     1.56250000000000e-02
     7.81250000000000e-03
     3.90625000000000e-03
     1.95312500000000e-03
     9.76562500000000e-04

*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "braid.h"

#include <petscts.h>

/*
 PETSc user functions
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = -x[0];
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat J, Mat B,void *ctx) {
  PetscScalar jac[1][1] = {{0}};
  PetscInt    rows[1] = {0},cols[1] = {0};

  PetscFunctionBeginUser;
  jac[0][0] = -1.0;
  MatSetValues(J,1,rows,1,cols,&jac[0][0],INSERT_VALUES);
  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  if (B!=J) {
    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
  }
  PetscFunctionReturn(0);
}

/*--------------------------------------------------------------------------
 * My integration routines
 *--------------------------------------------------------------------------*/

/* can put anything in my app and name it anything as well */
typedef struct _braid_App_struct
{
  MPI_Comm  comm;
  MPI_Comm  comm_t;
  MPI_Comm  comm_x;
  TS        ts;
  int       pt; /* number of processors in time */
  double    tstart;
  double    tstop;
  int       ntime;
  int       rank;
} my_App;

/* can put anything in my vector and name it anything as well */
typedef struct _braid_Vector_struct
{
  Vec vec;
} my_Vector;

int my_Step(braid_App app,braid_Vector ustop,braid_Vector fstop,braid_Vector u,braid_StepStatus status)
{
  double         tstart;             /* current time */
  double         tstop;              /* evolve u to this time*/
  PetscErrorCode ierr;
  /* Grab status of current time step */
  braid_StepStatusGetTstartTstop(status,&tstart,&tstop);

  /* set up ts */
  ierr = TSSetSolution(app->ts,u->vec);CHKERRQ(ierr);
  ierr = TSSetTime(app->ts,tstart);CHKERRQ(ierr);
  ierr = TSSetMaxTime(app->ts,tstop);CHKERRQ(ierr);
  ierr = TSSolve(app->ts,u->vec);CHKERRQ(ierr);
  /* no refinement */
//  braid_StepStatusSetRFactor(status,1);
  return 0;
}

int my_Residual(braid_App app,braid_Vector ustop,braid_Vector r,braid_StepStatus status)
{
  PetscScalar    *rvalue,*uvalue;
  double         tstart;             /* current time */
  double         tstop;              /* evolve to this time*/
  PetscErrorCode ierr;
  braid_StepStatusGetTstartTstop(status,&tstart,&tstop);

  ierr = VecGetArray(r->vec,&rvalue);CHKERRQ(ierr);
  ierr = VecGetArray(ustop->vec,&uvalue);CHKERRQ(ierr);
/* On the finest grid,each value is half the previous value */
  rvalue[0] = uvalue[0] - pow(0.5,tstop-tstart)*(rvalue[0]);
  ierr = VecRestoreArray(ustop->vec,&uvalue);CHKERRQ(ierr);
  ierr = VecRestoreArray(r->vec,&rvalue);CHKERRQ(ierr);
  return 0;
}

int my_Init(braid_App app,double t,braid_Vector *u_ptr)
{
  MPI_Comm       comm_x = app->comm_x;
  my_Vector      *u;
  PetscScalar    *uvalue;
  PetscErrorCode ierr;
  u = (my_Vector *) malloc(sizeof(my_Vector));
  ierr = VecCreate(comm_x,&u->vec);CHKERRQ(ierr);
  ierr = VecSetSizes(u->vec,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetUp(u->vec);CHKERRQ(ierr);

  ierr = VecGetArray(u->vec,&uvalue);CHKERRQ(ierr);
  if (t == app->tstart) {
    uvalue[0] = 1.0;
  } else {
    /* Initialize all other time points */
    uvalue[0] = 0.456;
  }
  ierr = VecRestoreArray(u->vec,&uvalue);CHKERRQ(ierr);
  *u_ptr = u;
//  ierr = TSSetSolution(app->ts,u->vec);CHKERRQ(ierr);
  return 0;
}

int my_Clone(braid_App app,braid_Vector u,braid_Vector *v_ptr)
{
  my_Vector      *v;
  PetscErrorCode ierr;
  v = (my_Vector *) malloc(sizeof(my_Vector));
  ierr = VecDuplicate(u->vec,&v->vec);CHKERRQ(ierr);
  ierr = VecCopy(u->vec,v->vec);CHKERRQ(ierr);
  *v_ptr = v;
  return 0;
}

int my_Free(braid_App app,braid_Vector u)
{
  PetscErrorCode ierr;
  ierr = VecDestroy(&(u->vec));CHKERRQ(ierr);
  free(u);
  return 0;
}

int my_Sum(braid_App app,double alpha,braid_Vector x,double beta,braid_Vector y)
{
  PetscErrorCode ierr;
  if (y->vec == x->vec) {
    Vec z;
    ierr = VecDuplicate(x->vec,&z);CHKERRQ(ierr);
    ierr = VecCopy(x->vec,z);CHKERRQ(ierr);
    ierr = VecAXPBY(y->vec,alpha,beta,z);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
  } else {
    ierr = VecAXPBY(y->vec,alpha,beta,x->vec);CHKERRQ(ierr);
  }
  return 0;
}

int my_SpatialNorm(braid_App app,braid_Vector u,double *norm_ptr)
{
  PetscErrorCode ierr;
  ierr = VecNorm(u->vec,NORM_2,norm_ptr);CHKERRQ(ierr);
  return 0;
}

int my_Access(braid_App app,braid_Vector u,braid_AccessStatus astatus)
{
  MPI_Comm   comm   = (app->comm);
  double     tstart = (app->tstart);
  double     tstop  = (app->tstop);
  int        ntime  = (app->ntime);
  int        index,myid;
  char       filename[255];
  FILE       *file;
  double     t;
  const PetscScalar    *uvalue;
  PetscErrorCode ierr;
  braid_AccessStatusGetT(astatus,&t);
  index = ((t-tstart) / ((tstop-tstart)/ntime) + 0.1);

  MPI_Comm_rank(comm,&myid);

  sprintf(filename,"%s.%07d.%05d","ex-01.out",index,myid);
  file = fopen(filename,"w");
  ierr = VecGetArrayRead(u->vec,&uvalue);CHKERRQ(ierr);
  fprintf(file,"%.14e\n",uvalue[0]);
  ierr = VecRestoreArrayRead(u->vec,&uvalue);CHKERRQ(ierr);
  fflush(file);
  fclose(file);
  return 0;
}

int my_BufSize(braid_App app,int *size_ptr,braid_BufferStatus bstatus)
{
  *size_ptr = sizeof(double);
  return 0;
}

int my_BufPack(braid_App app,braid_Vector u,void *buffer,braid_BufferStatus bstatus)
{
  double *dbuffer = buffer;
  PetscScalar    *uvalue;
  PetscErrorCode ierr;

  ierr = VecGetArray(u->vec,&uvalue);CHKERRQ(ierr);
  dbuffer[0] = uvalue[0];
  ierr = VecRestoreArray(u->vec,&uvalue);CHKERRQ(ierr);
  return 0;
}

int my_BufUnpack(braid_App app,void *buffer,braid_Vector *u_ptr,braid_BufferStatus bstatus)
{
  MPI_Comm   comm_x = app->comm_x;
  double    *dbuffer = buffer;
  my_Vector *u;
  PetscScalar    *uvalue;
  PetscErrorCode ierr;

  u = (my_Vector *) malloc(sizeof(my_Vector));
  ierr = VecCreate(comm_x,&u->vec);CHKERRQ(ierr);
  ierr = VecSetSizes(u->vec,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetUp(u->vec);CHKERRQ(ierr);
  ierr = VecGetArray(u->vec,&uvalue);CHKERRQ(ierr);
  uvalue[0] = dbuffer[0];
  ierr = VecRestoreArray(u->vec,&uvalue);CHKERRQ(ierr);
  *u_ptr = u;
  return 0;
}

/*--------------------------------------------------------------------------
 * Main driver
 *--------------------------------------------------------------------------*/

int main (int argc,char *argv[])
{
  braid_Core    core;
  my_App        *app;
  MPI_Comm      comm = MPI_COMM_WORLD;
  MPI_Comm      comm_x,comm_t;
  double        tstart,tstop;
  int           ntime,rank;

  int           max_levels = 2;
  int           nrelax     = 1;
  int           nrelax0    = -1;
  double        tol        = 1.0e-06;
  int           cfactor    = 2;
  int           max_iter   = 100;
  int           fmg        = 0;
  int           res        = 0;

  int           arg_index;
  Mat            J;
  PetscErrorCode ierr;
  /* Initialize MPI */
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* ntime time intervals with spacing 1 */
  comm   = MPI_COMM_WORLD;
  ntime  = 10;
  tstart = 0.0;
  tstop  = tstart + ntime/2;

  /* Parse command line */
  arg_index = 1;
  while (arg_index < argc)
  {
     if ( strcmp(argv[arg_index],"-help") == 0 )
     {
        int  myid;
        MPI_Comm_rank(comm,&myid);
        if ( myid == 0 )
        {
           printf("\n");
           printf("  -ml  <max_levels> : set max levels\n");
           printf("  -nu  <nrelax>     : set num F-C relaxations\n");
           printf("  -nu0 <nrelax>     : set num F-C relaxations on level 0\n");
           printf("  -tol <tol>        : set stopping tolerance\n");
           printf("  -cf  <cfactor>    : set coarsening factor\n");
           printf("  -mi  <max_iter>   : set max iterations\n");
           printf("  -fmg              : use FMG cycling\n");
           printf("  -res              : use my residual\n");
           printf("\n");
        }
        exit(1);
     }
     else if ( strcmp(argv[arg_index],"-ml") == 0 )
     {
        arg_index++;
        max_levels = atoi(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-nu") == 0 )
     {
        arg_index++;
        nrelax = atoi(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-nu0") == 0 )
     {
        arg_index++;
        nrelax0 = atoi(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-tol") == 0 )
     {
        arg_index++;
        tol = atof(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-cf") == 0 )
     {
        arg_index++;
        cfactor = atoi(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-mi") == 0 )
     {
        arg_index++;
        max_iter = atoi(argv[arg_index++]);
     }
     else if ( strcmp(argv[arg_index],"-fmg") == 0 )
     {
        arg_index++;
        fmg = 1;
     }
     else if ( strcmp(argv[arg_index],"-res") == 0 )
     {
        arg_index++;
        res = 1;
     }
     else
     {
        arg_index++;
        /*break;*/
     }
  }

  /* set up app structure */
  app = (my_App *) malloc(sizeof(my_App));
  app->tstart = tstart;
  app->tstop  = tstop;
  app->ntime  = ntime;
  app->rank   = rank;

  /* Create communicators for the time and space dimensions; serial in space and parallel in time */
  braid_SplitCommworld(&comm,1,&comm_x,&comm_t);
  app->comm = comm;
  app->comm_t = comm_t;
  app->comm_x = comm_x;

  /* Initialize PETSc */
  PETSC_COMM_WORLD = app->comm_x;
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);

  /*Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&app->ts);CHKERRQ(ierr);
  ierr = TSSetType(app->ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(app->ts,NULL,RHSFunction,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetType(J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(app->ts,J,J,RHSJacobian,NULL);CHKERRQ(ierr);
  ierr = TSSetTime(app->ts,tstart);CHKERRQ(ierr);
  ierr = TSSetMaxTime(app->ts,tstop);CHKERRQ(ierr);
  ierr = TSSetTimeStep(app->ts,(tstop-tstart)/ntime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(app->ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(app->ts);CHKERRQ(ierr);
  TSAdapt tsadapt;
  ierr = TSGetAdapt(app->ts,&tsadapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(tsadapt,TSADAPTNONE);CHKERRQ(ierr);
  /*
  Vec x;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetUp(x);CHKERRQ(ierr);
  ierr = TSSetSolution(app->ts,x);CHKERRQ(ierr);
  */
  braid_Init(comm,comm_t,tstart,tstop,ntime,app,
            my_Step,my_Init,my_Clone,my_Free,my_Sum,my_SpatialNorm,
            my_Access,my_BufSize,my_BufPack,my_BufUnpack,&core);

  braid_SetPrintLevel(core,2);
  braid_SetMaxLevels(core,max_levels);
//  braid_SetNRelax(core,-1,nrelax);
//  if (nrelax0 > -1)
//  {
//     braid_SetNRelax(core, 0,nrelax0);
//  }
  braid_SetAbsTol(core,tol);
  braid_SetCFactor(core,-1,cfactor);
  /*braid_SetCFactor(core, 0,10);*/
  braid_SetMaxIter(core,max_iter);
  if (fmg)
  {
    braid_SetFMG(core);
  }
  if (res)
  {
    braid_SetResidual(core,my_Residual);
  }

  braid_Drive(core);

  braid_Destroy(core);
  ierr = TSDestroy(&app->ts);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  free(app);
  PetscFinalize();
  /* Finalize MPI */
  MPI_Finalize();

  return (0);
}
