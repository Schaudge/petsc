
static char help[] = "Solves a tridiagonal linear system with KSP.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices              petscpc.h  - preconditioners
     petscis.h     - index sets
     petscviewer.h - viewers

  Note:  The corresponding parallel example is ex23.c
*/
#include <petscksp.h>
#include <petscdevice_cuda.h>
#include <cuda_profiler_api.h>
#include <petsc/private/deviceimpl.h>

__global__ static void MARKER_KERNEL() { }

int main(int argc,char **args)
{
  Vec            x, b, u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  PetscReal      norm;         /* norm of solution error */
  PetscInt       i,n = 10,col[3],its;
  PetscMPIInt    size;
  PetscScalar    value[3];

  PetscCallCUDA(cudaDeviceReset());
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  void *cublas_handle;
  PetscDeviceContext dctx;

  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_DEFAULT()));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextGetBLASHandle_Internal(dctx,&cublas_handle));
  PetscCall(PetscDeviceContextGetSOLVERHandle_Internal(dctx,&cublas_handle));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(PetscObjectSetName((PetscObject) x, "Solution"));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&b));
  PetscCall(VecDuplicate(x,&u));

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  PetscCall(MatCreate(PETSC_COMM_SELF,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Assemble matrix
  */
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    PetscCall(MatSetValues(A,1,&i,3,col,value,INSERT_VALUES));
  }
  i    = n - 1; col[0] = n - 2; col[1] = n - 1;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  PetscCall(MatSetValues(A,1,&i,2,col,value,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  PetscCall(VecSet(u,1.0));
  PetscCall(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_SELF,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the matrix that defines the preconditioner.
  */
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCJACOBI));
  PetscCall(KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt nwarmup = 2,nit = 1000;

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n_warmup",&nwarmup,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n_it",&nit,NULL));

  PetscLogStage warmup,timing;

  PetscCall(PetscLogStageRegister("Warmup",&warmup));
  PetscCall(PetscLogStageRegister("Timing",&timing));

  PetscCall(PetscLogStagePush(warmup));
  for (PetscInt i = 0; i < nwarmup; ++i) PetscCall(KSPSolve(ksp,b,x));
  PetscCall(PetscLogStagePop());

  PetscLogDouble *times;

  PetscCall(PetscMalloc1(nit,&times));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextSynchronize(NULL));
  PetscCallCUDA(cudaProfilerStart());
  for (PetscInt i = 0; i < nit; ++i) {
    PetscLogDouble begin,end;

    PetscCall(PetscLogStagePush(timing));
    MARKER_KERNEL<<<1,1,0,NULL>>>();
    PetscCall(PetscTime(&begin));
    PetscCall(KSPSolve(ksp,b,x));
    PetscCall(PetscTime(&end));
    MARKER_KERNEL<<<1,1,0,NULL>>>();
    PetscCall(PetscLogStagePop());
    times[i] = end-begin;
    PetscCallCUDA(cudaDeviceSynchronize());
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(PetscDeviceContextSynchronize(NULL));
  }
  PetscCallCUDA(cudaProfilerStop());

  PetscLogDouble tmin = PETSC_MAX_REAL,tmax = PETSC_MIN_REAL,ttotal = 0;
  for (PetscInt i = 0; i < nit; ++i) {
    ttotal += times[i];
    tmin = PetscMin(times[i],tmin);
    tmax = PetscMax(times[i],tmax);
  }
  PetscCall(PetscFree(times));

  KSPType type;

  PetscCall(KSPGetType(ksp,&type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"KSP type: '%s', nit %" PetscInt_FMT ", total time %gs, min %gs, max %gs, avg. %gs\n",type,nit,ttotal,tmin,tmax,ttotal/nit));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x,-1.0,u));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Norm of error %g, Iterations %" PetscInt_FMT "\n",(double)norm,its));

  /* check that KSP automatically handles the fact that the the new non-zero values in the matrix are propagated to the KSP solver */
  PetscCall(MatShift(A,2.0));
  PetscCall(KSPSolve(ksp,b,x));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}
