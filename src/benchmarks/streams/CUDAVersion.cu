/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

  It measures the memory system on the device.
  The implementation is in single precision.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <mpi.h>

#define N       20000000
#define NTIMES  50
#define OFFSET  0

#ifndef MIN
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x,y) ((x)>(y) ? (x) : (y))
#endif

#define CHKERRQ(ierr)   do {if(ierr) return ierr;} while(0)
#define CHKERRCUDA(err) do {if(err) return err;} while(0)

static double a[N+OFFSET],
              b[N+OFFSET],
              c[N+OFFSET];
static double *d_a, *d_b, *d_c;

static double bytes[4] = {
  2 * sizeof(double) * N,
  2 * sizeof(double) * N,
  3 * sizeof(double) * N,
  3 * sizeof(double) * N
};

__global__ void STREAM_Copy(double *a, const double *b, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Scale(double *a, const double *b, double scale,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = scale*b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Add(double *a, const double *b, const double *c,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = b[idx]+c[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Triad(double *a, const double *b, const double *c, double scalar, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = b[idx]+scalar*c[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

int main(int argc, char *argv[])
{
  int           ierr;
  cudaError_t   err;
  int           j,k,nthreads,nblocks;
  int           rank,size,devCount,device;
  double        irate[4],rate[4];
  float         times[4][NTIMES];
  double        mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
  double        scalar=3.0;
  cudaEvent_t   start,stop;
  size_t        sz;
  FILE          *fd;

  ierr = MPI_Init(&argc,&argv);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);

  err = cudaGetDeviceCount(&devCount);CHKERRCUDA(err);
  device = rank % devCount;
  err = cudaSetDevice(device);CHKERRCUDA(err);

  /* Configure block and grid */
  nthreads = 512;
  nblocks  = (N+nthreads-1)/nthreads;

  /* Init arrays on host and then copy them to device  */
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
  sz  = sizeof(double)*N;
  err = cudaMalloc((void**)&d_a,sz);CHKERRCUDA(err);
  err = cudaMalloc((void**)&d_b,sz);CHKERRCUDA(err);
  err = cudaMalloc((void**)&d_c,sz);CHKERRCUDA(err);
  err = cudaMemcpy(d_a,a,sz,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(d_b,b,sz,cudaMemcpyHostToDevice);CHKERRCUDA(err);
  err = cudaMemcpy(d_c,c,sz,cudaMemcpyHostToDevice);CHKERRCUDA(err);

  err = cudaDeviceSynchronize();CHKERRCUDA(err);

  /* Both timers report msec (10^-3 sec) */
  err = cudaEventCreate(&start);CHKERRCUDA(err);
  err = cudaEventCreate(&stop);CHKERRCUDA(err);

  for (k=0; k<NTIMES; ++k) {
    err = cudaEventRecord(start, 0);CHKERRCUDA(err);
    STREAM_Copy<<<nblocks,nthreads>>>(d_c, d_a, N);
    err = cudaEventRecord(stop, 0);CHKERRCUDA(err);
    err = cudaEventSynchronize(stop);CHKERRCUDA(err);
    err = cudaEventElapsedTime(&times[0][k], start, stop);CHKERRCUDA(err);

    err = cudaEventRecord(start, 0);CHKERRCUDA(err);
    STREAM_Scale<<<nblocks,nthreads>>>(d_b, d_c, scalar, N);
    err = cudaEventRecord(stop, 0);CHKERRCUDA(err);
    err = cudaEventSynchronize(stop);CHKERRCUDA(err);
    err = cudaEventElapsedTime(&times[1][k], start, stop);CHKERRCUDA(err);

    err = cudaEventRecord(start, 0);CHKERRCUDA(err);
    STREAM_Add<<<nblocks,nthreads>>>(d_c, d_a, d_b,  N);
    err = cudaEventRecord(stop, 0);CHKERRCUDA(err);
    err = cudaEventSynchronize(stop);CHKERRCUDA(err);
    err = cudaEventElapsedTime(&times[2][k], start, stop);CHKERRCUDA(err);

    err = cudaEventRecord(start, 0);CHKERRCUDA(err);
    STREAM_Triad<<<nblocks,nthreads>>>(d_a, d_b, d_c, scalar, N);
    err = cudaEventRecord(stop, 0);CHKERRCUDA(err);
    err = cudaEventSynchronize(stop);CHKERRCUDA(err);
    err = cudaEventElapsedTime(&times[3][k], start, stop);CHKERRCUDA(err);
  }

  err = cudaEventDestroy(stop);CHKERRCUDA(err);
  err = cudaEventDestroy(start);CHKERRCUDA(err);
  err = cudaFree(d_a);CHKERRCUDA(err);
  err = cudaFree(d_b);CHKERRCUDA(err);
  err = cudaFree(d_c);CHKERRCUDA(err);

  /*   --- SUMMARY --- */
  for (k=0; k<NTIMES; k++)
    for (j=0; j<4; j++) mintime[j] = MIN(mintime[j], times[j][k]);

  for (j=0; j<4; j++) irate[j] = 1.0E-03*bytes[j]/mintime[j]; /* mintime is in msec */
  ierr = MPI_Reduce(irate,rate,4,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);CHKERRQ(ierr);

  if (!rank) {
    if (size == 1) {
      printf("%d %11.1f   Rate (MB/s)\n",size, rate[3]);
      fd = fopen("flops","w");
      fprintf(fd,"%g\n",rate[3]);
      fclose(fd);
    } else {
      double prate;
      fd = fopen("flops","r");
      fscanf(fd,"%lg",&prate);
      fclose(fd);
      printf("%d %11.1f   Rate (MB/s) %g \n", size, rate[3],rate[3]/prate);
    }
  }

  MPI_Finalize();
  return 0;
}
