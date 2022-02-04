#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include <mpi.h>

#define CHKERRQ(ierr) assert(ierr == hipSuccess)

__global__ void Pack(int N,const double *vec_d,double *buf_d)
{
  int i = hipBlockIdx_x*hipBlockDim_x+hipThreadIdx_x;
  if(i<N) {
    buf_d[i]   = vec_d[i];
    buf_d[i+N] = vec_d[i];
  }
}

int main(int argc,char **argv)
{
  hipError_t  ierr = hipSuccess;
  const int   vecLen = 7;
  const int   bufLen = 2*vecLen;
  double      vec_h[vecLen],buf_h[bufLen];
  double      *buf_d,*vec_d;

  MPI_Init(&argc,&argv);
  ierr = hipMalloc(&vec_d,vecLen*2*sizeof(double));CHKERRQ(ierr);
  ierr = hipMalloc(&buf_d,bufLen*sizeof(double));CHKERRQ(ierr);

  for (int k=0; k<100; k++) {
    for (int i=0; i<vecLen; i++) {
      memset(vec_h,0,vecLen*sizeof(double));
      vec_h[i] = 1.0;
      ierr = hipMemcpyAsync(vec_d,vec_h,vecLen*sizeof(double),hipMemcpyHostToDevice,NULL);CHKERRQ(ierr);
      hipLaunchKernelGGL(Pack,dim3((vecLen+255)/256),dim3(256),0,0,vecLen,vec_d,buf_d);
      ierr = hipMemcpyAsync(buf_h,buf_d,bufLen*sizeof(double),hipMemcpyDeviceToHost,NULL);CHKERRQ(ierr);
      ierr = hipStreamSynchronize(NULL);CHKERRQ(ierr);
      assert(buf_h[i] == 1.0 && buf_h[i+vecLen] == 1.0);
    }
  }
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
    test:
      suffix: debug5
    test:
      suffix: debug6
    test:
      suffix: debug7
    test:
      suffix: debug8
    test:
      suffix: debug9
    test:
      suffix: debug10
TEST*/
