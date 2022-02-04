#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <mpi.h>
#include <hipblas.h>

#define CHKERRQ(ierr) assert(ierr == hipSuccess)

#define CHKBLAS(ierr) assert(ierr == HIPBLAS_STATUS_SUCCESS)


__global__ void Pack(int N,const double *vec_d,double *buf_d)
{
  int i = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
  if(i<N) {
    buf_d[i]   = vec_d[i];
    buf_d[i+N] = vec_d[i];
  }
}

void CPU_fill(double *v,int m, int n)
{
  for(int i=0;i<m*n;i++)
    v[i] = i;
}

int main(int argc,char **argv)
{
  hipError_t  ierr = hipSuccess;
  hipblasStatus_t stat = HIPBLAS_STATUS_SUCCESS;
  const int   vecLen = 7;
  const int   bufLen = 2*vecLen;
  double      vec_h[vecLen],buf_h[bufLen];
  double      *buf_d,*vec_d;

  hipblasHandle_t handle;
  double          *A_h,*B_h,*C_h;
  double          *A_d,*B_d,*C_d;
  int             m = 8;
  const double    alf = 1.0f;
  const double    bet = 0.0f;
  const double    *alpha = &alf;
  const double    *beta = &bet;


  MPI_Init(&argc,&argv);

  stat = hipblasCreate(&handle);CHKBLAS(stat);

  // Allocate 3 arrays on CPU
  A_h = (double *)malloc(m*m*sizeof(double));
  B_h = (double *)malloc(m*m*sizeof(double));
  C_h = (double *)malloc(m*m*sizeof(double));
  CPU_fill(A_h, m, m);
  CPU_fill(B_h, m, m);
  CPU_fill(C_h, m, m);

  // Allocate 3 arrays on GPU
  ierr = hipMalloc(&A_d, m*m*sizeof(double));CHKERRQ(ierr);
  ierr = hipMalloc(&B_d, m*m*sizeof(double));CHKERRQ(ierr);
  ierr = hipMalloc(&C_d, m*m*sizeof(double));CHKERRQ(ierr);
  ierr = hipMemcpy(A_d,A_h,m*m*sizeof(double),hipMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = hipMemcpy(B_d,B_h,m*m*sizeof(double),hipMemcpyHostToDevice);CHKERRQ(ierr);
  ierr = hipMemcpy(C_d,C_h,m*m*sizeof(double),hipMemcpyHostToDevice);CHKERRQ(ierr);

  ierr = hipMalloc(&vec_d,vecLen*2*sizeof(double));CHKERRQ(ierr);
  ierr = hipMalloc(&buf_d,bufLen*sizeof(double));CHKERRQ(ierr);

  for (int k=0; k<1; k++) {
    for (int i=0; i<vecLen; i++) {
      memset(vec_h,0,vecLen*sizeof(double));
      vec_h[i] = 1.0;
      ierr = hipMemcpyAsync(vec_d,vec_h,vecLen*sizeof(double),hipMemcpyHostToDevice,NULL);CHKERRQ(ierr);
      hipLaunchKernelGGL(Pack,dim3((vecLen+255)/256),dim3(256),0,0,vecLen,vec_d,buf_d);
      ierr = hipMemcpyAsync(buf_h,buf_d,bufLen*sizeof(double),hipMemcpyDeviceToHost,NULL);CHKERRQ(ierr);
      ierr = hipStreamSynchronize(NULL);CHKERRQ(ierr);
      double sum = 0.0;
      for (int j=0; j<bufLen; j++) {
        sum += buf_h[j];
      }
      if (sum != 2.0) {
        printf("Error in Pack\n");
        exit(1);
      }
      stat = hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, m, m, alpha, A_d, m, B_d, m, beta, C_d, m);CHKBLAS(stat);
    }
  }

  free(A_h);
  free(B_h);
  free(C_h);
  ierr = hipFree(A_d);CHKERRQ(ierr);
  ierr = hipFree(B_d);CHKERRQ(ierr);
  ierr = hipFree(C_d);CHKERRQ(ierr);
  stat = hipblasDestroy(handle);CHKBLAS(stat);

  ierr = hipFree(vec_d);CHKERRQ(ierr);
  ierr = hipFree(buf_d);CHKERRQ(ierr);
  MPI_Finalize();
  return ierr;
}

/*TEST

  testset:
    nsize: 3
    requires: hip
    output_file: output/ex255_1.out

    test:
      suffix: debug1
    test:
      suffix: debug2
    test:
      suffix: debug3
    test:
      suffix: debug4
    
TEST*/
