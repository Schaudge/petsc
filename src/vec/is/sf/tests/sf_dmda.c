static char help[] = "Tests DMGlobalToLocal/DMLocalToGlobal\n\n";

#include <petscdm.h>
#include <petscdmda.h>

#include <petsccublas.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h>

/* Same values as OSU microbenchmark */
#define LAT_LOOP_SMALL 10000
#define LAT_SKIP_SMALL 100
#define LAT_LOOP_LARGE 1000
#define LAT_SKIP_LARGE 10

#define LARGE_MESSAGE_SIZE 8192

int main(int argc,char **argv)
{
  PetscErrorCode   ierr;
  cudaError_t      cerr;
  DM               da[64];
  Vec              l[64],g[64];
  PetscLogDouble   t_start=0,t_end=0,time[64];
  DMBoundaryType   bx,by;
  DMDAStencilType  stype = DMDA_STENCIL_STAR;
  PetscInt         i,j,n,N; /* n=size of the subdomain; N=3*n=size of the domain */
  PetscInt         minn=4,maxn=4096,m=3;
  PetscInt         dof=1,swidth=1,nskip,niter;
  PetscMPIInt      rank,size;
  PetscInt         nprof=128; /* the n where we do profiling */
  PetscInt         msgsize;
  char             istring[128];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  bx = by = DM_BOUNDARY_PERIODIC;
  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-nprof",&nprof,NULL);CHKERRQ(ierr);

  for (n=minn,i=0; n<=maxn; n*=2,i++) {
    /* Create distributed array and get vectors */
    N    = 3*n;
    ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,N,N,m,m,dof,swidth,NULL,NULL,&da[i]);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da[i]);CHKERRQ(ierr);
    ierr = DMSetUp(da[i]);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da[i],&g[i]);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da[i],&l[i]);CHKERRQ(ierr);
  }

  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.category = 1000;

  nskip = LAT_SKIP_SMALL;
  niter = LAT_LOOP_SMALL;
  for (n=minn,j=0; n<=maxn; n*=2,j++) {
    ierr = VecSet(g[j],1.0);CHKERRQ(ierr);
    msgsize = n*sizeof(PetscScalar);

    if (msgsize > LARGE_MESSAGE_SIZE) {
      nskip = LAT_SKIP_LARGE;
      niter = LAT_LOOP_LARGE;
    }
    ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

    for (i=0; i<niter + nskip; i++) {
      if (i == nskip) {
        cerr    = cudaDeviceSynchronize();CHKERRCUDA(cerr);
        if (n == nprof) {cerr = cudaProfilerStart();CHKERRCUDA(cerr);}
        ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
        t_start = MPI_Wtime();
      }
      if (n == nprof) {
        sprintf(istring,"iter-%d",i);
        eventAttrib.message.ascii = istring;
        nvtxMarkEx(&eventAttrib);
      }
      ierr = DMGlobalToLocalBegin(da[j],g[j],INSERT_VALUES,l[j]);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(da[j],g[j],INSERT_VALUES,l[j]);CHKERRQ(ierr);
      ierr = DMLocalToGlobalBegin(da[j],l[j],ADD_VALUES,g[j]);CHKERRQ(ierr);
      ierr = DMLocalToGlobalEnd(da[j],l[j],ADD_VALUES,g[j]);CHKERRQ(ierr);
    }
    cerr    = cudaDeviceSynchronize();CHKERRCUDA(cerr);
    ierr    = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
    t_end   = MPI_Wtime();
    time[j] = (t_end - t_start)*1e6 / (niter*2);
    if (n == nprof) {cerr = cudaProfilerStop();CHKERRCUDA(cerr);}
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t##  PetscSF DMDA2d Global<-->Local test ##\n \tSubgrid size n\t\tLatency(us)\n");CHKERRQ(ierr);
  for (n=minn,j=0; n<=maxn; n*=2,j++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%16D \t %16.4f\n",n,time[j]);CHKERRQ(ierr);
    ierr = VecDestroy(&l[j]);CHKERRQ(ierr);
    ierr = VecDestroy(&g[j]);CHKERRQ(ierr);
    ierr = DMDestroy(&da[j]);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return 0;
}
