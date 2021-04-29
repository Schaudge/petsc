static const char help[] = "PetscSF Ping-pong test\n\n";

#include <petscconf.h>
#include <petscsys.h>
#include <petscsf.h>
#include <unistd.h>

#if defined(PETSC_HAVE_CUDA)
  #include <petsccublas.h>
  #include <petsc/private/cudavecimpl.h> /* For PetscNvshmemMalloc(), which is not supposed to be used by users */
  #include <cuda_profiler_api.h>

  #define CUDA_DEVICE_SYNC() do {cudaError_t cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);} while(0)
  #define CUDA_STREAM_SYNC() do {cudaError_t cerr = cudaStreamSynchronize(NULL);CHKERRCUDA(cerr);} while(0)
  #define PROFILING_START()  do {cudaError_t cerr = cudaProfilerStart();CHKERRCUDA(cerr);} while(0)
  #define PROFILING_STOP()   do {cudaError_t cerr = cudaProfilerStop();CHKERRCUDA(cerr);} while(0)
#else
  #define CUDA_DEVICE_SYNC() 0
  #define CUDA_STREAM_SYNC() 0
  #define PROFILING_START()  0
  #define PROFILING_STOP()   0
#endif

PetscInt LAT_SKIP_SMALL     = 100;
PetscInt LAT_SKIP_LARGE     = 10;
PetscInt LAT_LOOP_SMALL     = 10000;
PetscInt LAT_LOOP_LARGE     = 1000;
PetscInt LARGE_MESSAGE_SIZE = 8192;

// PetscInt LAT_SKIP_SMALL     = 10;
// PetscInt LAT_SKIP_LARGE     = 10;
// PetscInt LAT_LOOP_SMALL     = 180;
// PetscInt LAT_LOOP_LARGE     = 180;
// PetscInt LARGE_MESSAGE_SIZE = 8192;

PETSC_STATIC_INLINE PetscErrorCode PetscMallocWithMemType(PetscMemType mtype,size_t size,void** ptr)
{
  PetscErrorCode ierr;
  unsigned long  align_size = sysconf(_SC_PAGESIZE);

  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {ierr = posix_memalign(ptr,align_size,size);CHKERRQ(ierr);} /* page-aligned as in OSU */
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_CUDA) {
    cudaError_t cerr;
    ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
    cerr = cudaMalloc(ptr,size);CHKERRCUDA(cerr);
  }
#endif
#if defined(PETSC_HAVE_NVSHMEM)
  else if (mtype == PETSC_MEMTYPE_NVSHMEM) {ierr = PetscNvshmemMalloc(size,ptr);CHKERRQ(ierr);}
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unsupported mtype = %d",(int)mtype);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFreeWithMemType(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {free(ptr);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_CUDA) {cudaError_t cerr = cudaFree(ptr);CHKERRCUDA(cerr);}
#endif
#if defined(PETSC_HAVE_NVSHMEM)
  else if (mtype == PETSC_MEMTYPE_NVSHMEM) {PetscErrorCode ierr = PetscNvshmemFree(ptr);CHKERRQ(ierr);}
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unsupported mtype = %d",(int)mtype);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscMemsetWithMemType(PetscMemType mtype,void* ptr,int c, size_t n)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {memset(ptr,c,n);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_CUDA) {cudaError_t cerr = cudaMemset(ptr,c,n);CHKERRCUDA(cerr);}
  else if (mtype == PETSC_MEMTYPE_NVSHMEM) {cudaError_t cerr = cudaMemset(ptr,c,n);CHKERRCUDA(cerr);}
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unsupported mtype = %d",(int)mtype);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  PetscSF        sf[64];
  PetscLogDouble t_start=0,t_end=0,time[64];
  PetscInt       i,j,n,nroots,nleaves,niter=100,nskip=10;
  PetscInt       maxn=512*1024; /* max 4M bytes messages */
  PetscSFNode    *iremote;
  PetscMPIInt    rank,size;
  PetscScalar    *sbuf=NULL,*rbuf=NULL;
  size_t         msgsize;
  PetscMemType   mtype = PETSC_MEMTYPE_HOST;
  char           mstring[16]={0};
  PetscBool      isHost=PETSC_FALSE,isCuda=PETSC_FALSE,isNvshmem=PETSC_FALSE,set;
  PetscInt       profile_message_size = -1; /* Do profile at this message size. negative means no profiling */
  PetscInt       profile_loop_count = 20; /* Do this many iterations in profiling if enabled */

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (size != 2) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"This test has to be run with two MPI ranks\n");CHKERRQ(ierr);
    MPI_Abort(PETSC_COMM_WORLD,-1);
  }

  ierr = PetscMalloc1(maxn,&iremote);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-profile_message_size",&profile_message_size,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-profile_loop_count",&profile_loop_count,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-mtype",mstring,16,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PetscStrcasecmp(mstring,"host",&isHost);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(mstring,"cuda",&isCuda);CHKERRQ(ierr);
    if (!isCuda) {ierr = PetscStrcasecmp(mstring,"device",&isCuda);CHKERRQ(ierr);}; /* alias for 'cuda' */
    ierr = PetscStrcasecmp(mstring,"nvshmem",&isNvshmem);CHKERRQ(ierr);

    if (isCuda)         mtype = PETSC_MEMTYPE_CUDA;
    else if (isNvshmem) mtype = PETSC_MEMTYPE_NVSHMEM;
    else if (isHost)    mtype = PETSC_MEMTYPE_HOST;
    else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unkonwn memtype: %s\n",mstring);
  }

  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&sbuf);CHKERRQ(ierr);
  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&rbuf);CHKERRQ(ierr);

  for (n=1,i=0; n<=maxn; n*=2,i++) {
    ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf[i]);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(sf[i]);CHKERRQ(ierr);
    if (!rank) {
      nroots  = n;
      nleaves = 0;
    } else {
      nroots  = 0;
      nleaves = n;
      for (j=0; j<nleaves; j++) {
        iremote[j].rank  = 0;
        iremote[j].index = j;
      }
    }
    ierr = PetscSFSetGraph(sf[i],nroots,nleaves,NULL,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);
  }

  ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

  for (n=1,j=0; n<=maxn; n*=2,j++) {
    msgsize = sizeof(PetscScalar)*n; /* 8n is the message size */
    ierr = PetscMemsetWithMemType(mtype,sbuf,'a',msgsize);CHKERRQ(ierr);
    ierr = PetscMemsetWithMemType(mtype,rbuf,'b',msgsize);CHKERRQ(ierr);

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    } else {
      nskip = LAT_SKIP_SMALL;
      niter = LAT_LOOP_SMALL;
    }

    /* Overwrite the loop count when profiling is enabled */
    if (msgsize == profile_message_size) {
      nskip = 10;
      niter = profile_loop_count;
    }

    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

    for (i=0; i<niter + nskip; i++) {
      if (i == nskip) {
        cerr    = cudaDeviceSynchronize();CHKERRCUDA(cerr);
        if (msgsize == profile_message_size) {
          cerr = cudaProfilerStart();CHKERRCUDA(cerr);
        }
        ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
        t_start = MPI_Wtime();
      }
      ierr = PetscSFBcastWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,sbuf,mtype,rbuf,MPI_REPLACE);CHKERRQ(ierr); /* rank 0->1, root->leaf*/
      ierr = PetscSFBcastEnd(sf[j],MPIU_SCALAR,sbuf,rbuf,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceWithMemTypeBegin(sf[j],MPIU_SCALAR,mtype,sbuf,mtype,rbuf,MPI_REPLACE);CHKERRQ(ierr); /* rank 1->0, leaf->root */
      ierr = PetscSFReduceEnd(sf[j],MPIU_SCALAR,sbuf,rbuf,MPI_REPLACE);CHKERRQ(ierr);
    }
    cerr = cudaDeviceSynchronize();CHKERRCUDA(cerr);

    ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start)*1e6 / (niter*2);

    if (msgsize == profile_message_size) {
      cerr = cudaProfilerStop();CHKERRCUDA(cerr);
    }
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t##  PetscSF Ping-pong test on %s ##\n  Message(Bytes) \t\tLatency(us)\n", mtype==PETSC_MEMTYPE_HOST? "Host" : mstring);CHKERRQ(ierr);
  for (n=1,j=0; n<=maxn; n*=2,j++) {
    ierr = PetscSFDestroy(&sf[j]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%16D \t %16.4f\n",sizeof(PetscScalar)*n,time[j]);CHKERRQ(ierr);
  }

  ierr = PetscFreeWithMemType(mtype,sbuf);CHKERRQ(ierr);
  ierr = PetscFreeWithMemType(mtype,rbuf);CHKERRQ(ierr);
  ierr = PetscFree(iremote);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/**TEST
   test:
     nsize: 2
     args: -mtype host

   test:
     nsize: 2
     suffix: 2
     requires: cuda
     args: -mtype device
TEST**/

