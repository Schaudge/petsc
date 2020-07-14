/*
   Implements the sequential libaxb vectors for hybrid architectures.
*/

#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <libaxb.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqaxb/vecaxbimpl.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscAXBFlag for the vector
    Does NOT zero the hybrid array
 */
PetscErrorCode VecAXBAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  Vec_AXB     *vecaxb;

  PetscFunctionBegin;
  if (!v->spptr) {
    ierr = PetscMalloc(sizeof(Vec_AXB),&v->spptr);CHKERRQ(ierr);
    vecaxb = (Vec_AXB*)v->spptr;
    ierr = axbVecCreateBegin(axb_handle, &vecaxb->vec);CHKERRQ(ierr);
    ierr = axbVecSetSize(vecaxb->vec,v->map->n);CHKERRQ(ierr);
    ierr = axbVecSetMemBackend(vecaxb->vec,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbVecSetOpBackend(vecaxb->vec,axb_op_backend);CHKERRQ(ierr);
    ierr = axbVecCreateEnd(vecaxb->vec);CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecAXBCopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  Vec_AXB     *vecaxb = (Vec_AXB*)v->spptr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  ierr = VecAXBAllocateCheck(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(VEC_AXBCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    ierr = axbVecSetValues(vecaxb->vec,((Vec_Seq*)v->data)->array,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_AXBCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}


/*
     VecAXBCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecAXBCopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  Vec_AXB     *vecaxb;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  ierr = VecAXBAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr               = PetscLogEventBegin(VEC_AXBCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    vecaxb          = (Vec_AXB*)v->spptr;
    ierr               = axbVecGetValues(vecaxb->vec,((Vec_Seq*)v->data)->array,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    ierr               = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscLogEventEnd(VEC_AXBCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}


/*MC
   VECSEQAXB - VECSEQAXB = "seqaxb" - The basic sequential vector, modified to use libaxb (for GPUs, threads, etc.)

   Options Database Keys:
. -vec_type seqaxb - sets the vector type to VECSEQAXB during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

PetscErrorCode VecAYPX_SeqAXB(Vec yin,PetscScalar alpha,Vec xin)
{
  const struct axbVec_s *xaxb;
  struct axbVec_s       *yaxb;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBGetArray(yin,&yaxb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = axbVecCopy(xaxb,yaxb);CHKERRQ(ierr);
  } else {
    struct axbScalar_s *alpha_axb;

    ierr = axbScalarCreate(axb_handle,&alpha_axb,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbVecAYPX(yaxb,alpha_axb,xaxb);CHKERRQ(ierr);
    if (alpha == (PetscScalar)1.0) {
      ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    }
  }
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArray(yin,&yaxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqAXB(Vec yin,PetscScalar alpha,Vec xin)
{
  const struct axbVec_s *xaxb;
  struct axbVec_s       *yaxb;
  struct axbScalar_s    *alpha_axb;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (alpha != (PetscScalar)0.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_axb,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecAXPY(yaxb,alpha_axb,xaxb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqAXB(Vec win, Vec xin, Vec yin)
{
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xaxb=NULL,*yaxb=NULL;
  struct axbVec_s       *waxb=NULL;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecAXBGetArrayWrite(win,&waxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(yin,&yaxb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecPointwiseDivide(waxb,xaxb,yaxb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(yin,&yaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayWrite(win,&waxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqAXB(Vec win,Vec xin,Vec yin)
{
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xaxb=NULL,*yaxb=NULL;
  struct axbVec_s       *waxb=NULL;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecAXBGetArrayWrite(win,&waxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(yin,&yaxb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecPointwiseMult(waxb,xaxb,yaxb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(yin,&yaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayWrite(win,&waxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqAXB(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const struct axbVec_s *xaxb=NULL,*yaxb=NULL;
  struct axbVec_s       *waxb=NULL;
  struct axbScalar_s    *alpha_axb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = axbVecCopy(yaxb,waxb);CHKERRQ(ierr);
  } else {
    ierr = axbScalarCreate(axb_handle,&alpha_axb,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(yin,&yaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArrayWrite(win,&waxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecWAXPY(waxb,alpha_axb,xaxb,yaxb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(yin,&yaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayWrite(win,&waxb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqAXB(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode           ierr;
  PetscInt                 n = xin->map->n,j;
  struct axbVec_s          *xaxb;
  struct axbScalar_s       **alphas_axb;
  const struct axbVec_s    **yin_axb;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nv,&yin_axb);CHKERRQ(ierr);
  ierr = PetscMalloc1(nv,&alphas_axb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayWrite(xin,&xaxb);CHKERRQ(ierr);
  for (j=0; j<nv; ++j) {
    ierr = axbScalarCreate(axb_handle,&(alphas_axb[j]),(void*)(alpha + j),AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(y[j],&(yin_axb[j]));CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = axbVecMAXPY(xaxb,nv,(const struct axbScalar_s * const *)alphas_axb,yin_axb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  for (j=0; j<nv; ++j) {
    ierr = VecAXBRestoreArrayRead(y[j],&(yin_axb[j]));CHKERRQ(ierr);
    ierr = axbScalarDestroy(alphas_axb[j]);
  }
  ierr = VecAXBRestoreArrayWrite(xin,&xaxb);CHKERRQ(ierr);
  ierr = PetscFree(yin_axb);CHKERRQ(ierr);
  ierr = PetscFree(alphas_axb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqAXB(Vec xin,Vec yin,PetscScalar *z)
{
  const struct axbVec_s *xaxb=NULL,*yaxb=NULL;
  struct axbScalar_s    *alpha_axb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)z,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(yin,&yaxb);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecDot(xaxb,yaxb,alpha_axb);CHKERRQ(ierr);
  ierr = axbScalarGetValue(alpha_axb,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(yin,&yaxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecMDot_SeqAXB(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode        ierr;
  PetscInt              i,n = xin->map->n;
  const struct axbVec_s **yin_axb,*xaxb;
  struct axbScalar_s    **alpha_axb;

  PetscFunctionBegin;
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqAXB not positive.");
  if (xin->map->n > 0) {
    ierr = PetscMalloc1(nv,&yin_axb);CHKERRQ(ierr);
    ierr = PetscMalloc1(nv,&alpha_axb);CHKERRQ(ierr);
    for (i=0; i<nv; ++i) {
      ierr = axbScalarCreate(axb_handle,&(alpha_axb[i]),(void*)(z+i),AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
      ierr = VecAXBGetArrayRead(yin[i],&yin_axb[i]);CHKERRQ(ierr);
    }
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = axbVecMDot(xaxb,nv,yin_axb,alpha_axb);CHKERRQ(ierr);
    for (i=0; i<nv; ++i) {
      ierr = axbScalarGetValue(alpha_axb[i],(void*)(z+i),AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = axbScalarDestroy(alpha_axb[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(yin_axb);CHKERRQ(ierr);
    ierr = PetscFree(alpha_axb);CHKERRQ(ierr);
  } else { /* local size zero */
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode VecSet_SeqAXB(Vec xin,PetscScalar alpha)
{
  struct axbVec_s    *xaxb;
  struct axbScalar_s *alpha_axb;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecAXBGetArrayWrite(xin,&xaxb);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecSet(xaxb, alpha_axb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayWrite(xin,&xaxb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqAXB(Vec xin,PetscScalar alpha)
{
  struct axbVec_s    *xaxb;
  struct axbScalar_s *alpha_axb;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqAXB(xin,alpha);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayWrite(xin,&xaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecScale(xaxb, alpha_axb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayWrite(xin,&xaxb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqAXB(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqAXB(xin,yin,z);CHKERRQ(ierr); /* Note: This needs to be changed as soon as VecAXB supports complex arithmetic */
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqAXB(Vec xin,Vec yin)
{
  const struct axbVec_s *xaxb;
  struct axbVec_s       *yaxb;
  PetscErrorCode        ierr;
  PetscBool             copy_on_gpu = PETSC_FALSE;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) copy_on_gpu = PETSC_TRUE;
    else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      if (yin->offloadmask == PETSC_OFFLOAD_GPU || yin->offloadmask == PETSC_OFFLOAD_BOTH) copy_on_gpu = PETSC_TRUE;
    } 

    if (copy_on_gpu) {
      ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
      ierr = VecAXBGetArrayWrite(yin,&yaxb);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      ierr = axbVecCopy(xaxb,yaxb);CHKERRQ(ierr);
      ierr = WaitForGPU();CHKERRQ(ierr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
      ierr = VecAXBRestoreArrayWrite(yin,&yaxb);CHKERRQ(ierr);
    } else {
      ierr = VecCopy_SeqAXB_Private(xin,yin);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqAXB(Vec xin,Vec yin)
{
  struct axbVec_s *xaxb,*yaxb;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecAXBGetArray(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecSwap(xaxb,yaxb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecAXBRestoreArray(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArray(yin,&yaxb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqAXB(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscScalar           a = alpha,b = beta;
  const struct axbVec_s *xaxb;
  struct axbVec_s       *yaxb;
  struct axbScalar_s    *alpha_axb,*beta_axb;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_SeqAXB(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqAXB(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqAXB(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecCopy(xaxb,yaxb);CHKERRQ(ierr);
    ierr = axbVecScale(yaxb,alpha_axb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
  } else {
    ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbScalarCreate(axb_handle,&beta_axb,(void*)&beta,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBGetArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecScale(yaxb,beta_axb);CHKERRQ(ierr);
    ierr = axbVecAXPY(yaxb,alpha_axb,xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArray(yin,&yaxb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
    ierr = axbScalarDestroy(beta_axb);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqAXB(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqAXB(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqAXB(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqAXB(zin,gamma);CHKERRQ(ierr);
    ierr = VecAXPY_SeqAXB(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqAXB(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqAXB(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode        ierr;
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xaxb;
  struct axbScalar_s    *alpha_axb;

  PetscFunctionBegin;
  if (n > 0) {
    ierr = VecAXBGetArrayRead(xin,&xaxb);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbScalarCreate(axb_handle,&alpha_axb,(void*)z,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      ierr = axbVecNorm2(xaxb,alpha_axb);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_axb,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) {
      ierr = axbVecNormInf(xaxb,alpha_axb);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_axb,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    } else if (type == NORM_1) {
      ierr = axbVecNorm1(xaxb,alpha_axb);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_axb,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    } else if (type == NORM_1_AND_2) {
      ierr = VecNorm_SeqAXB(xin,NORM_1,z);CHKERRQ(ierr);
      ierr = VecNorm_SeqAXB(xin,NORM_2,z+1);CHKERRQ(ierr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_axb);CHKERRQ(ierr);
    ierr = VecAXBRestoreArrayRead(xin,&xaxb);CHKERRQ(ierr);
  } else {
    z[0] = 0.0;
    if (type == NORM_1_AND_2) z[1] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqAXB(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode        ierr;
  PetscReal             n=s->map->n;
  const struct axbVec_s *saxb,*taxb;
  struct axbScalar_s    *st_axb,*norm_axb;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&st_axb,(void*)dp,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = axbScalarCreate(axb_handle,&norm_axb,(void*)nm,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(s,&saxb);CHKERRQ(ierr);
  ierr = VecAXBGetArrayRead(t,&taxb);CHKERRQ(ierr);
  ierr = axbVecDotNorm2(saxb,taxb,st_axb,norm_axb);CHKERRQ(ierr);
  ierr = axbScalarGetValue(st_axb,(void*)dp,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = axbScalarGetValue(norm_axb,(void*)nm,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(s,&saxb);CHKERRQ(ierr);
  ierr = VecAXBRestoreArrayRead(t,&taxb);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqAXB(Vec v)
{
  PetscErrorCode ierr;
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  Vec_AXB     *vecaxb = (Vec_AXB*)v->spptr;

  PetscFunctionBegin;
  if (v->spptr) {
    if (vecaxb->vec) {
      ierr = axbVecDestroy(vecaxb->vec);CHKERRQ(ierr);
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs) {
    if (vs->array_allocated) { ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr); }
    ierr = PetscFree(vs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecConjugate_SeqAXB(Vec xin)
{
  PetscFunctionBegin;
  /* Add code here when adding support for complex arithmetic */
  PetscFunctionReturn(0);
}


/*@C
   VecAXBGetArray - Provides access to the axb buffer (a axbVec_t from libaxb) inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecAXBGetArray() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The axbVec_t has to be released by calling
   VecAXBRestoreArray().  Upon restoring the vector data
   the data on the host will be marked as out of date.  A subsequent
   access of the host data will thus incur a data transfer from the
   device to the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the internal axbVec_t object providing hybrid capabilities

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecAXBRestoreArray(), VecAXBGetArrayRead(), VecAXBGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBGetArray(Vec v, struct axbVec_s **vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  *vec   = 0;
  ierr = VecAXBCopyToGPU(v);CHKERRQ(ierr);
  *vec   = ((Vec_AXB*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecAXBRestoreArray - Restore a libaxb device object previously acquired with VecAXBGetArray().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecAXBRestoreArray() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecAXBGetArray(), VecAXBGetArrayRead(), VecAXBGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBRestoreArray(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecAXBGetArrayRead - Provides read access to the hybrid buffer (a axbVec_t from libaxb) inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecAXBGetArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The returned pointer has to be released by calling
   VecAXBRestoreArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the internal axbVec_t object providing hybrid capabilities.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecAXBRestoreArrayRead(), VecAXBGetArray(), VecAXBGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBGetArrayRead(Vec v, const struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  *a   = 0;
  ierr = VecAXBCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_AXB*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecAXBRestoreArrayRead - Restore a hybrid buffer (a axbVec_t from libaxb) previously acquired with VecAXBGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecAXBRestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecAXBGetArrayRead(), VecAXBGetArrayWrite(), VecAXBGetArray(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBRestoreArrayRead(Vec v, const struct axbVec_s **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecAXBGetArrayWrite - Provides write access to the hybrid buffer (a axbVec_t from libaxb) inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The returned pointer needs to be released with
   VecAXBRestoreArrayWrite().  When the pointer is released the
   host data of the vector is marked as out of data.  Subsequent access
   of the host data with e.g. VecGetArray() incurs a device to host data
   transfer.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the internal axbVec_t object providing hybrid capabilities

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecAXBRestoreArrayWrite(), VecAXBGetArray(), VecAXBGetArrayRead(), VecAXBGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBGetArrayWrite(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  *a   = 0;
  ierr = VecAXBAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_AXB*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecAXBRestoreArrayWrite - Restore a hybrid buffer (a axbVec_t from libaxb) previously acquired with VecAXBGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecAXBRestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecAXBGetArrayWrite(), VecAXBGetArray(), VecAXBGetArrayRead(), VecAXBGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecAXBRestoreArrayWrite(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQAXB,VECMPIAXB);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

