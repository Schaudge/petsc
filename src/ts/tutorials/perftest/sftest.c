
/*
       Reads in an adjacency graph for communication between ranks
         -fname petscbinaryfilewithsparsematrix

       Creates unique artificial indices for the pack from global and unpack into local

         -num_fields <num> multiples the amount of data indicated by communication matrix by num (0 is a special case)

       Runs the communication for timing

       Saves result by appending to the file sftest_out
         -fout alternativenameforsftest_out

       See ReadInSftest.m for post-processing the results of runs with sftest.c
*/
#include <petscmat.h>
#include <petscsf.h>
#if defined(PETSC_HAVE_CUDA)
#include <petsccublas.h>
#include <cuda_profiler_api.h>
#endif

/*
    Creates a random number from rstart to rend, exclusive, will not repeat a previously created random number
*/
PetscInt RandomInt(PetscInt rstart,PetscInt rend,PetscRandom rand,PetscBool *filled)
{
  PetscReal r;
  PetscInt  id;

  while (PETSC_TRUE) {
    PetscRandomGetValue(rand,&r);
    id = rstart + (PetscInt)PetscFloorReal((rend - rstart)*r);
    if (!filled[id]) {
      filled[id] = PETSC_TRUE;
      return id;
    }
  }
}

int main(int argc, char **argv)
{
  MPI_Comm              comm;
  PetscMPIInt		rank,size;
  Vec                   local,global;
  VecScatter            sf;
  PetscInt              n,nlocal,ncols,m,*globalindices,*localindices,N;
  PetscInt              i,j,cnt,niter=150,nskip=10;
  IS                    islocal,isglobal;
  const PetscInt        *cols,*ranges;
  const PetscScalar     *vals;
  PetscErrorCode        ierr;
  char                  fname[PETSC_MAX_PATH_LEN],fout[PETSC_MAX_PATH_LEN];
  PetscReal             num_fields=1,tStart=0,tEnd,tTotal; /* Silence -Wmaybe-uninitialized */
  Mat                   A;
  PetscViewer           viewer;
  PetscLogEvent         event;
  PetscBool             *localindicesused,*globalindicesused;
  PetscBool             profile = PETSC_FALSE,matSym = PETSC_FALSE;
  PetscRandom           rand;
  PetscClassId          classid;
  PetscLogStage         stage;
#if defined(PETSC_HAVE_CUDA)
  cudaError_t           cerr;
#endif
  PetscBool             flg;
  const PetscInt        HEADER = 1211213;

  ierr = PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "true");CHKERRQ(ierr);
  ierr = PetscInitialize(&argc, &argv,(char *) 0, NULL);if(ierr){ return ierr;}
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Sftest Options","");CHKERRQ(ierr);
  {
    ierr = PetscArrayzero(fname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscOptionsString("-fname","Matrix filename","PetscViewerBinaryOpen()",fname,fname,sizeof(fname),&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(comm,PETSC_ERR_USER,"You must run with -fname file where file contains a PETSc sparse matrix of the same dimension as the number of ranks");
    ierr = PetscStrcpy(fout,"sftest_out");CHKERRQ(ierr);
    ierr = PetscOptionsString("-fout","Output filename","PetscViewerBinaryOpen()",fout,fout,sizeof(fout),&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-num_fields","Number of fields to communicate (0 indicates 1 data item per connection)","",num_fields,&num_fields,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-symmetric","Treat communication matrix as symmetric, i.e. A = A^T+A","",matSym,&matSym,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-cuda_profile","Enable CUDA profiling","",profile,&profile,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-niter","Number of iterations in timing loop","",niter,&niter,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,fname,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = MatViewFromOptions(A,NULL,"-mat_view_input");CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerCreate(comm,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_APPEND);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,fout);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);

  if (matSym) {
    Mat At;

    ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
    ierr = MatAXPY(A,1.0,At,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatDestroy(&At);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A,NULL,"-mat_view_symmetric");CHKERRQ(ierr);
  }

  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  if (m != size) SETERRQ(comm,PETSC_ERR_ARG_SIZ,"Matrix must have one row per rank");
  if (num_fields == 0.0) {
    PetscInt          nz;
    const PetscScalar *vals;
    const PetscInt    *cols;
    Mat               B;

    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatGetRow(B,rank,&nz,&cols,&vals);CHKERRQ(ierr);
    for (PetscInt i=0; i<nz; i++) {
      if (vals[i] != 0.0) {
        ierr = MatSetValue(A,rank,cols[i],16.0,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatRestoreRow(B,rank,&nz,&cols,&vals);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    ierr = MatScale(A,num_fields);CHKERRQ(ierr);
  }

  /* make 10 % of the local portion of the global vector be sent */
  ierr = MatGetRow(A,rank,&ncols,&cols,&vals);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<ncols; i++) cnt += vals[i]/sizeof(PetscScalar);
  n = PetscMax(1000,10*cnt);
  ierr = MatRestoreRow(A,rank,&ncols,&cols,&vals);CHKERRQ(ierr);

  ierr = VecCreate(comm,&global);CHKERRQ(ierr);
  ierr = VecSetFromOptions(global);CHKERRQ(ierr);
  ierr = VecSetSizes(global,n,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecViewFromOptions(global,NULL,"-vec_view_global");CHKERRQ(ierr);
  ierr = VecGetSize(global,&N);CHKERRQ(ierr);
  ierr = PetscCalloc1(N,&globalindicesused);CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
  /* for each rank you recieve from make up as indices for all the locations they will send to you and where you will put them*/
  ierr = VecGetOwnershipRanges(global,&ranges);CHKERRQ(ierr);
  /* If the matrix isn't symmetric then transposing it will create new allocations */
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatGetRow(A,rank,&ncols,&cols,&vals);CHKERRQ(ierr);

  /* make 10 % of the local vector be ghost points */
  cnt = 0;
  for (i=0; i<ncols; i++) cnt += vals[i]/sizeof(PetscScalar);
  nlocal = PetscMax(1000,10*cnt);

  ierr = VecCreate(PETSC_COMM_SELF,&local);CHKERRQ(ierr);
  ierr = VecSetFromOptions(local);CHKERRQ(ierr);
  ierr = VecSetSizes(local,nlocal,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecViewFromOptions(local,NULL,"-vec_view_local");CHKERRQ(ierr);
  ierr = PetscCalloc1(nlocal,&localindicesused);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlocal,&localindices);CHKERRQ(ierr);
  ierr = PetscMalloc1(nlocal,&globalindices);CHKERRQ(ierr);

  cnt = 0;
  for (i=0; i<ncols; i++) {
    for (j=0; j<vals[i]/sizeof(PetscScalar); j++) {
      globalindices[cnt]  = RandomInt(ranges[cols[i]+1],ranges[cols[i]],rand,globalindicesused);
      localindices[cnt++] = RandomInt(0,nlocal,rand,localindicesused);
    }
  }
  ierr = MatRestoreRow(A,rank,&ncols,&cols,&vals);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(globalindicesused);CHKERRQ(ierr);
  ierr = PetscFree(localindicesused);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,cnt,globalindices,PETSC_OWN_POINTER,&isglobal);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,cnt,localindices,PETSC_OWN_POINTER,&islocal);CHKERRQ(ierr);

  ierr = ISViewFromOptions(isglobal,NULL,"-is_view_global");CHKERRQ(ierr);
  ierr = ISViewFromOptions(islocal,NULL,"-is_view_local");CHKERRQ(ierr);
  ierr = VecScatterCreate(global,isglobal,local,islocal,&sf);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(sf,NULL,"-vecscatter_view");CHKERRQ(ierr);
  ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);CHKERRQ(ierr);
  ierr = VecScatterBegin(sf,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sf,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("GlobalToLocal",&stage);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("GlobalToLocal",&classid);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("GlobalToLocal",classid, &event);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  for (i=0; i<niter+nskip; i++) {
    if (i == nskip) {
#if defined(PETSC_HAVE_CUDA)
      cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
      if (profile) {cerr = cudaProfilerStart();CHKERRCUDA(cerr);}
#endif
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      tStart = MPI_Wtime();
    }
    ierr = VecScatterBegin(sf,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sf,global,local,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_CUDA)
  cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);
#endif
  tEnd = MPI_Wtime();
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (profile) {cerr = cudaProfilerStop();CHKERRCUDA(cerr);}
#endif
  tTotal = ((tEnd-tStart)*((float)1e6))/((float)niter);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&tTotal,1,MPI_DOUBLE,MPI_MAX,comm);CHKERRMPI(ierr);
  ierr = PetscSFGetUseNVSHMEM(sf,&flg);CHKERRQ(ierr);

  ierr = PetscPrintf(comm,"Time %g num_fields %g %s\n",tTotal,num_fields,flg ? "Using NVSHMEM" : "Not using NVSHMEM");CHKERRQ(ierr);

  ierr = PetscViewerBinaryWrite(viewer,&HEADER,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&tTotal,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&HEADER,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&num_fields,1,PETSC_DOUBLE);CHKERRQ(ierr);
  /* use real to indicate if NVSHMEM is used since Bool is not supported loading from Matlab */
  num_fields = flg ? 1.0 : 0.0;
  ierr = PetscViewerBinaryWrite(viewer,&HEADER,1,PETSC_INT);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&num_fields,1,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
