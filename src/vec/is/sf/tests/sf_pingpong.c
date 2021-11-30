static const char help[] = "PetscSF new Ping-pong test\n\n";

#include <petscsf.h>
#include <unistd.h>
#include <petscdevice.h>

/* Same values as OSU microbenchmark */
#define LAT_LOOP_SMALL 10000
#define LAT_SKIP_SMALL 100
#define LAT_LOOP_LARGE 1000
#define LAT_SKIP_LARGE 10
#define LARGE_MESSAGE_SIZE 8192

PETSC_STATIC_INLINE PetscErrorCode PetscMallocWithMemType(PetscMemType mtype,size_t size,void** ptr)
{
  PetscErrorCode ierr;
  unsigned long  align_size = sysconf(_SC_PAGESIZE);

  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {ierr = posix_memalign(ptr,align_size,size);CHKERRQ(ierr);} /* page-aligned as in OSU */
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_CUDA) {cudaError_t cerr = cudaMalloc(ptr,size);CHKERRCUDA(cerr);}
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFreeWithMemType_Private(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {free(ptr);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) {cudaError_t cerr = cudaFree(ptr);CHKERRCUDA(cerr);}
#endif
  PetscFunctionReturn(0);
}

/* Free memory and set ptr to NULL when succeeded */
#define PetscFreeWithMemType(t,p) ((p) && (PetscFreeWithMemType_Private((t),(p)) || ((p)=NULL,0)))

PETSC_STATIC_INLINE PetscErrorCode PetscMemsetWithMemType(PetscMemType mtype,void* ptr,int c, size_t n)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {memset(ptr,c,n);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) {cudaError_t cerr = cudaMemset(ptr,c,n);CHKERRCUDA(cerr);}
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscMemcpyFromHostWithMemType(PetscMemType mtype,void* dst, const void *src, size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {ierr = PetscMemcpy(dst,src,n);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) {
    cudaError_t cerr = cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  }
#endif
  PetscFunctionReturn(0);
}


int main(int argc,char **argv)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;
  PetscSF        sf[64];
  PetscLogDouble t_start=0,t_end=0,time[64];
  PetscInt       i,j,n,nroots,nleaves,niter=100,nskip=10;
  PetscInt       maxn=512*1024; /* max 4M bytes messages */
  PetscSFNode    *iremote;
  PetscMPIInt    rank,size;
  PetscScalar    *rootdata=NULL,*leafdata=NULL,*pbuf,*ebuf;
  size_t         msgsize;
  PetscMemType   mtype = PETSC_MEMTYPE_HOST;
  char           mstring[16]={0};
  PetscBool      flg,set;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxn,&iremote);CHKERRQ(ierr);

  ierr = PetscOptionsGetString(NULL,NULL,"-mtype",mstring,16,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PetscStrcasecmp(mstring,"cuda",&flg);CHKERRQ(ierr);
    if (flg) mtype = PETSC_MEMTYPE_DEVICE;
    else {
      ierr = PetscStrcasecmp(mstring,"host",&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Unkonwn memtype: %s\n",mstring);
      mtype = PETSC_MEMTYPE_HOST;
    }
  }

  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&rootdata);CHKERRQ(ierr);
  ierr = PetscMallocWithMemType(mtype,sizeof(PetscScalar)*maxn,(void**)&leafdata);CHKERRQ(ierr);

  ierr = PetscMalloc2(maxn,&pbuf,maxn,&ebuf);CHKERRQ(ierr);
  for (i=0; i<maxn; i++) {
    pbuf[i] = 3.142;
    ebuf[i] = 2.718;
  }

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
    ierr = PetscSFSetUp(sf[i]);CHKERRQ(ierr);
  }

  nskip = LAT_SKIP_SMALL;
  niter = LAT_LOOP_SMALL;
  for (n=1,j=0; n<=maxn; n*=2,j++) {
    msgsize = sizeof(PetscScalar)*n;
    ierr = PetscMemcpyFromHostWithMemType(mtype,rootdata,pbuf,msgsize);CHKERRQ(ierr);
    ierr = PetscMemcpyFromHostWithMemType(mtype,leafdata,ebuf,msgsize);CHKERRQ(ierr);

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    }
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);

    for (i=0; i<niter + nskip; i++) {
      if (i == nskip) {
        cerr    = cudaDeviceSynchronize();CHKERRCUDA(cerr);
        ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
        t_start = MPI_Wtime();
      }
      ierr = PetscSFBcastBegin(sf[j],MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr); /* rank 0->1, root->leaf*/
      ierr = PetscSFBcastEnd(sf[j],MPIU_SCALAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(sf[j],MPIU_SCALAR,leafdata,rootdata,MPI_REPLACE);CHKERRQ(ierr); /* rank 1->0, leaf->root */
      ierr = PetscSFReduceEnd(sf[j],MPIU_SCALAR,leafdata,rootdata,MPI_REPLACE);CHKERRQ(ierr);
    }
    cerr    = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start)*1e6 / (niter*2);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t##  PetscSF new Ping-pong test on %s ##\n  Message(Bytes) \t\tLatency(us)\n", mtype==PETSC_MEMTYPE_HOST? "Host" : "Device");CHKERRQ(ierr);
  for (n=1,j=0; n<=maxn; n*=2,j++) {
    ierr = PetscSFDestroy(&sf[j]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%16D \t %16.4f\n",sizeof(PetscScalar)*n,time[j]);CHKERRQ(ierr);
  }

  ierr = PetscFreeWithMemType(mtype,rootdata);CHKERRQ(ierr);
  ierr = PetscFreeWithMemType(mtype,leafdata);CHKERRQ(ierr);
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
     requires: cuda
     args: -mtype cuda
TEST**/
