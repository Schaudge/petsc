static char help[]="Testing PETSC MatMult\n\n";

#include <petscmat.h>

#if defined(PETSC_HAVE_CUDA)
#include <petsccublas.h>
#include <cuda_profiler_api.h>

#define CUDA_STREAM_SYNC() do {cudaError_t cerr = cudaStreamSynchronize(NULL);CHKERRCUDA(cerr);} while(0)
#define PROFILING_START()  do {cudaError_t cerr = cudaProfilerStart();CHKERRCUDA(cerr);} while(0)
#define PROFILING_STOP()   do {cudaError_t cerr = cudaProfilerStart();CHKERRCUDA(cerr);} while(0)
#else
#define CUDA_STREAM_SYNC() 0
#define PROFILING_START()  0
#define PROFILING_STOP()   0
#endif

static PetscErrorCode FileNameStripExt(char *fname)
{
  char *end;

  PetscFunctionBegin;
  end = fname + strlen(fname);
  while (end > fname && *end != '.') --end;
  if (end > fname) *end = '\0';
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  Mat             A;
  PetscViewer     fd;
  char            matfile[PETSC_MAX_PATH_LEN],vecfile[PETSC_MAX_PATH_LEN];
  PetscBool       flg,set,doVerify=PETSC_FALSE,pretest=PETSC_FALSE;
  Vec             x,y,refy;
  PetscLogStage   stage;
  PetscInt        i,slen,niter=1000,nskip=10,rstart,rend;
  MatInfo         info;
  PetscLogDouble  t1=0,t2,etime;
  PetscRandom     rctx;
  PetscReal       norm;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Load the matrix from a binary file */
  ierr = PetscOptionsGetString(NULL,NULL,"-f",matfile,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,matfile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* Create vecs to do y = Ax */
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&refy);CHKERRQ(ierr);

  /* See if we just need to generate x, y for latter use */
  ierr = PetscOptionsGetBool(NULL,NULL,"-pretest",&pretest,&set);CHKERRQ(ierr);

  if (pretest) {
    ierr = PetscOptionsGetString(NULL,NULL,"-xy",vecfile,PETSC_MAX_PATH_LEN,&set);CHKERRQ(ierr);
    if (!set) { /* If user did not provide the vector file name */
      vecfile[0] = 0;
      ierr = PetscStrcpy(vecfile,matfile);CHKERRQ(ierr);
      ierr = FileNameStripExt(vecfile);CHKERRQ(ierr);
      slen = strlen(vecfile);
      ierr = PetscSNPrintf(vecfile+slen,PETSC_MAX_PATH_LEN-slen,".xy");CHKERRQ(ierr);
    }
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecfile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = MatMult(A,x,refy);CHKERRQ(ierr);
    ierr = VecView(x,fd);CHKERRQ(ierr);
    ierr = VecView(refy,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
    goto finish;
  }

  /* Load x, y from a vecfile if provided */
  ierr = PetscOptionsGetString(NULL,NULL,"-xy",vecfile,PETSC_MAX_PATH_LEN,&doVerify);CHKERRQ(ierr);
  if (doVerify) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,vecfile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = VecLoad(x,fd);CHKERRQ(ierr);
    ierr = VecLoad(refy,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else {
    ierr = VecSet(x,1.0);CHKERRQ(ierr);
  }

  /* Read options -niter, -nskip */
  ierr = PetscOptionsGetInt(NULL,NULL,"-niter",&niter,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nskip",&nskip,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetInfo(A,MAT_LOCAL,&info);CHKERRQ(ierr);
  //ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Proc %d has %.1f nonzero, and row: %d ~ %d\n", rank, info.nz_used, rstart, rend-1);CHKERRQ(ierr);
  //ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("MatMult", &stage);CHKERRQ(ierr);
  PROFILING_START();
  for (i=0; i<niter + nskip; i++) {
    if (i == nskip) {
      CUDA_STREAM_SYNC();
      ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
      ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
      t1   = MPI_Wtime();
    }
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
  }
  CUDA_STREAM_SYNC();

  ierr  = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
  t2    = MPI_Wtime();
  etime = (t2- t1)*1e6/niter;
  ierr  = PetscLogStagePop();CHKERRQ(ierr);
  PROFILING_STOP();

  if (doVerify) {
    ierr = VecAYPX(y,-1,refy);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
    if (norm > 1e-1) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"MatMult wrong with result %g", (double)norm);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average MatMult time = %16.4f, with norm %g\n",etime, (double)norm);CHKERRQ(ierr);

finish:
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&refy);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
