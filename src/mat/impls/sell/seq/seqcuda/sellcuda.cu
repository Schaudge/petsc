#include <cuda_runtime.h>

#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#include <../src/mat/impls/sell/seq/sell.h>  /*I   "petscmat.h"  I*/

typedef struct {
  PetscInt  *colidx;           /* column index */
  MatScalar *val;
  PetscInt  *sliidx;
  PetscInt  nonzerostate;
} Mat_SeqSELLCUDA;

static PetscErrorCode MatSeqSELLCUDA_Destroy(Mat_SeqSELLCUDA **cudastruct)
{
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (*cudastruct) {
    if ((*cudastruct)->colidx) {
      cerr = cudaFree((*cudastruct)->colidx);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->val) {
      cerr = cudaFree((*cudastruct)->val);CHKERRCUDA(cerr);
    }
    if ((*cudastruct)->sliidx) {
      cerr = cudaFree((*cudastruct)->sliidx);CHKERRCUDA(cerr);
    }
    ierr = PetscFree(*cudastruct);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSeqSELLCUDACopyToGPU(Mat A)
{
  Mat_SeqSELLCUDA  *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  Mat_SeqSELL      *a = (Mat_SeqSELL*)A->data;
  PetscErrorCode   ierr;
  cudaError_t      cerr;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(MAT_CUDACopyToGPU,A,0,0,0);CHKERRQ(ierr);
    if (!A->assembled || A->nonzerostate != cudastruct->nonzerostate) {
      if (cudastruct->colidx) {
        cerr = cudaFree(cudastruct->colidx);CHKERRCUDA(cerr);
      }
      if (cudastruct->val) {
        cerr = cudaFree(cudastruct->val);CHKERRCUDA(cerr);
      }
      if (cudastruct->sliidx) {
        cerr = cudaFree(cudastruct->sliidx);CHKERRCUDA(cerr);
      }
      cerr = cudaMalloc((void **)&(cudastruct->sliidx),(a->totalslices+1)*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = cudaMalloc((void **)&(cudastruct->colidx),a->maxallocmat*sizeof(PetscInt));CHKERRCUDA(cerr);
      cerr = cudaMalloc((void **)&(cudastruct->val),a->maxallocmat*sizeof(MatScalar));CHKERRCUDA(cerr);
    }
    /* copy values, nz or maxallocmat? */
    cerr = cudaMemcpy(cudastruct->sliidx,a->sliidx,(a->totalslices+1)*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(cudastruct->colidx,a->colidx,a->sliidx[a->totalslices]*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(cudastruct->val,a->val,a->sliidx[a->totalslices]*sizeof(MatScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = PetscLogCpuToGpu(a->sliidx[a->totalslices]*(sizeof(MatScalar)+sizeof(PetscInt))+(a->totalslices+1)*sizeof(PetscInt));CHKERRQ(ierr);
    cudastruct->nonzerostate = A->nonzerostate;
    cerr  = WaitForGPU();CHKERRCUDA(cerr);
    A->offloadmask = PETSC_OFFLOAD_BOTH;
    ierr = PetscLogEventEnd(MAT_CUDACopyToGPU,A,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

__global__ void matmult_seqsell_basic_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt  i,row,slice_id,row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) y[row] = sum;
      } else {
        y[row] = sum;
      }
    }
  }
}

__global__ void matmultadd_seqsell_basic_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  PetscInt  i,row,slice_id,row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i=sliidx[slice_id]+row_in_slice; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) z[row] = y[row] + sum;
      } else {
        z[row] = y[row] + sum;
      }
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,PetscScalar *y)
{
  __shared__ MatScalar shared[256];
  PetscInt   i,row,slice_id,row_in_slice;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    if (blockDim.y > 4) {
      __syncthreads();
      if (threadIdx.y < 4) {
        shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
      }
    }
    if (blockDim.y > 2) {
      __syncthreads();
      if (threadIdx.y < 2) {
        shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
      }
    }
    if (blockDim.y > 1) {
      __syncthreads();
      if (threadIdx.y < 1) {
        shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      }
    }
    if (threadIdx.y < 1) {
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) y[row] = shared[threadIdx.x];
      } else {
        y[row] = shared[threadIdx.x];
      }
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel(PetscInt nrows,PetscInt totalslices,const PetscInt *acolidx,const MatScalar *aval,const PetscInt *sliidx,const PetscScalar *x,const PetscScalar *y,PetscScalar *z)
{
  __shared__ MatScalar shared[256];
  PetscInt   i,row,slice_id,row_in_slice;
  /* one thread per row. */
  row = blockIdx.x*blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row/SLICE_HEIGHT;
    row_in_slice = row%SLICE_HEIGHT;

    shared[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
    for (i=sliidx[slice_id]+row_in_slice+SLICE_HEIGHT*threadIdx.y; i<sliidx[slice_id+1]; i+=SLICE_HEIGHT*blockDim.y) shared[threadIdx.y*blockDim.x+threadIdx.x] += aval[i] * x[acolidx[i]];
    if (blockDim.y > 4) {
      __syncthreads();
      if (threadIdx.y < 4) {
        shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+4)*blockDim.x+threadIdx.x];
      }
    }
    if (blockDim.y > 2) {
      __syncthreads();
      if (threadIdx.y < 2) {
        shared[threadIdx.y*blockDim.x+threadIdx.x] += shared[(threadIdx.y+2)*blockDim.x+threadIdx.x];
      }
    }
    if (blockDim.y > 1) {
      __syncthreads();
      if (threadIdx.y < 1) {
        shared[threadIdx.x] += shared[blockDim.x+threadIdx.x];
      }
    }
    if (threadIdx.y < 1) {
      if (slice_id == totalslices-1 && nrows%SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows%SLICE_HEIGHT)) z[row] = y[row] + shared[threadIdx.x];
      } else {
        z[row] = y[row] + shared[threadIdx.x];
      }
    }
  }
}

PetscErrorCode MatMult_SeqSELLCUDA(Mat A,Vec xx,Vec yy)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt          totalslices=a->totalslices,nrows = A->rmap->n;
  MatScalar         *aval = cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx = cudastruct->sliidx;
  PetscErrorCode    ierr;
  cudaError_t       cerr;
  PetscInt          nblocks,blocksize = 256;

  PetscFunctionBegin;
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  nblocks = (nrows+blocksize-1)/blocksize;
  if (nblocks >= 80) {
    matmult_seqsell_basic_kernel<<<nblocks,blocksize>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
  } else {
    PetscInt avg_width;
    dim3     block1(256,1),block2(128,2),block4(64,4),block8(32,8);
    avg_width = a->sliidx[a->totalslices]/(SLICE_HEIGHT*a->totalslices);
    if (avg_width > 64) {
      matmult_seqsell_tiled_kernel<<<nblocks*8,block8>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
    } else if (avg_width > 32) {
      matmult_seqsell_tiled_kernel<<<nblocks*4,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
    } else if (avg_width > 16) {
      matmult_seqsell_tiled_kernel<<<nblocks*2,block2>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
    } else {
      matmult_seqsell_tiled_kernel<<<nblocks,block1>>>(nrows,totalslices,acolidx,aval,sliidx,x,y);
    }
  }
  cerr = WaitForGPU();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqSELLCUDA(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA*)A->spptr;
  PetscScalar       *z;
  const PetscScalar *y,*x;
  PetscInt          totalslices=a->totalslices,nrows = A->rmap->n;
  MatScalar         *aval=cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx = cudastruct->sliidx;
  PetscErrorCode    ierr;
  cudaError_t       cerr;

  PetscFunctionBegin;
  ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  if (a->nz) {
    PetscInt nblocks,blocksize = 256;
    ierr = VecCUDAGetArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(zz,&z);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    nblocks = (nrows+blocksize-1)/blocksize;
    if (nblocks >= 80) {
      matmultadd_seqsell_basic_kernel<<<nblocks,blocksize>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
    } else {
      PetscInt avg_width;
      dim3     block1(256,1),block2(128,2),block4(64,4),block8(32,8);
      avg_width = a->sliidx[a->totalslices]/(SLICE_HEIGHT*a->totalslices);
      if (avg_width > 64) {
        matmultadd_seqsell_tiled_kernel<<<nblocks*8,block8>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
      } else if (avg_width > 32) {
        matmultadd_seqsell_tiled_kernel<<<nblocks*4,block4>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
      } else if (avg_width > 16) {
        matmultadd_seqsell_tiled_kernel<<<nblocks*2,block2>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
      } else {
        matmultadd_seqsell_tiled_kernel<<<nblocks,block1>>>(nrows,totalslices,acolidx,aval,sliidx,x,y,z);
      }
    }
    cerr = WaitForGPU();CHKERRCUDA(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xx,&x);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yy,&y);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(zz,&z);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*a->nz);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(yy,zz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_SeqSELLCUDA(PetscOptionItems *PetscOptionsObject,Mat A)
{
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SeqSELLCUDA options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

static PetscErrorCode MatAssemblyEnd_SeqSELLCUDA(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_SeqSELL(A,mode);CHKERRQ(ierr);
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  if (A->factortype == MAT_FACTOR_NONE) {
    ierr = MatSeqSELLCUDACopyToGPU(A);CHKERRQ(ierr);
  }
  A->ops->mult    = MatMult_SeqSELLCUDA;
  A->ops->multadd = MatMultAdd_SeqSELLCUDA;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqSELLCUDA(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->factortype==MAT_FACTOR_NONE) {
    if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) {
      ierr = MatSeqSELLCUDA_Destroy((Mat_SeqSELLCUDA**)&A->spptr);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy_SeqSELL(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqSELLCUDA(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat             C;
  Mat_SeqSELLCUDA *cudastruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqSELL(A,cpvalues,B);CHKERRQ(ierr);
  C    = *B;
  ierr = PetscFree(C->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&C->defaultvectype);CHKERRQ(ierr);

  /* inject CUSPARSE-specific stuff */
  if (C->factortype==MAT_FACTOR_NONE) {
    ierr = PetscNew(&cudastruct);CHKERRQ(ierr);
    C->spptr = cudastruct;
  }

  C->ops->assemblyend    = MatAssemblyEnd_SeqSELLCUDA;
  C->ops->destroy        = MatDestroy_SeqSELLCUDA;
  C->ops->setfromoptions = MatSetFromOptions_SeqSELLCUDA;
  C->ops->mult           = MatMult_SeqSELLCUDA;
  C->ops->multadd        = MatMultAdd_SeqSELLCUDA;
  C->ops->duplicate      = MatDuplicate_SeqSELLCUDA;

  ierr = PetscObjectChangeTypeName((PetscObject)C,MATSEQSELLCUDA);CHKERRQ(ierr);
  C->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat B)
{
  Mat_SeqSELLCUDA *cudastruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFree(B->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&B->defaultvectype);CHKERRQ(ierr);

  /* inject CUSPARSE-specific stuff */
  if (B->factortype==MAT_FACTOR_NONE) {
    ierr = PetscNew(&cudastruct);CHKERRQ(ierr);
    B->spptr = cudastruct;
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqSELLCUDA;
  B->ops->destroy        = MatDestroy_SeqSELLCUDA;
  B->ops->setfromoptions = MatSetFromOptions_SeqSELLCUDA;
  B->ops->mult           = MatMult_SeqSELLCUDA;
  B->ops->multadd        = MatMultAdd_SeqSELLCUDA;
  B->ops->duplicate      = MatDuplicate_SeqSELLCUDA;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQSELLCUDA);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELLCUDA(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_SeqSELL(B);CHKERRQ(ierr);
  ierr = MatConvert_SeqSELL_SeqSELLCUDA(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
