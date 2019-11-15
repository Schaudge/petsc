/*
   Implements the sequential hybrid vectors.
*/

#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <libaxb.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqhybrid/vechybridimpl.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscHybridFlag for the vector
    Does NOT zero the hybrid array
 */
PetscErrorCode VecHybridAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  Vec_Hybrid     *vechybrid;

  PetscFunctionBegin;
  if (!v->spptr) {
    ierr = PetscMalloc(sizeof(Vec_Hybrid),&v->spptr);CHKERRQ(ierr);
    vechybrid = (Vec_Hybrid*)v->spptr;
    ierr = axbVecCreateBegin(axb_handle, &vechybrid->vec);CHKERRQ(ierr);
    ierr = axbVecSetSize(vechybrid->vec,v->map->n);CHKERRQ(ierr);
    ierr = axbVecSetMemBackend(vechybrid->vec,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbVecSetOpBackend(vechybrid->vec,axb_op_backend);CHKERRQ(ierr);
    ierr = axbVecCreateEnd(vechybrid->vec);CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecHybridCopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  Vec_Hybrid     *vechybrid = (Vec_Hybrid*)v->spptr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  ierr = VecHybridAllocateCheck(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    ierr = PetscLogEventBegin(VEC_HybridCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    ierr = axbVecSetValues(vechybrid->vec,((Vec_Seq*)v->data)->array,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_HybridCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}


/*
     VecHybridCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecHybridCopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  Vec_Hybrid     *vechybrid;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  ierr = VecHybridAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr               = PetscLogEventBegin(VEC_HybridCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    vechybrid          = (Vec_Hybrid*)v->spptr;
    ierr               = axbVecGetValues(vechybrid->vec,((Vec_Seq*)v->data)->array,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    ierr               = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr               = PetscLogEventEnd(VEC_HybridCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}


/*MC
   VECSEQHYBRID - VECSEQHYBRID = "seqhybrid" - The basic sequential vector, modified to use libaxb (for GPUs, threads, etc.)

   Options Database Keys:
. -vec_type seqhybrid - sets the vector type to VECSEQHYBRID during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

PetscErrorCode VecAYPX_SeqHybrid(Vec yin,PetscScalar alpha,Vec xin)
{
  const struct axbVec_s *xhybrid;
  struct axbVec_s       *yhybrid;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArray(yin,&yhybrid);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = axbVecCopy(xhybrid,yhybrid);CHKERRQ(ierr);
  } else {
    struct axbScalar_s *alpha_hybrid;

    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbVecAYPX(yhybrid,alpha_hybrid,xhybrid);CHKERRQ(ierr);
    if (alpha == (PetscScalar)1.0) {
      ierr = PetscLogGpuFlops(1.0*yin->map->n);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    }
  }
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArray(yin,&yhybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqHybrid(Vec yin,PetscScalar alpha,Vec xin)
{
  const struct axbVec_s *xhybrid;
  struct axbVec_s       *yhybrid;
  struct axbScalar_s    *alpha_hybrid;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (alpha != (PetscScalar)0.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecAXPY(yhybrid,alpha_hybrid,xhybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqHybrid(Vec win, Vec xin, Vec yin)
{
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xhybrid=NULL,*yhybrid=NULL;
  struct axbVec_s       *whybrid=NULL;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecHybridGetArrayWrite(win,&whybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecPointwiseDivide(whybrid,xhybrid,yhybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayWrite(win,&whybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqHybrid(Vec win,Vec xin,Vec yin)
{
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xhybrid=NULL,*yhybrid=NULL;
  struct axbVec_s       *whybrid=NULL;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecHybridGetArrayWrite(win,&whybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecPointwiseMult(whybrid,xhybrid,yhybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayWrite(win,&whybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqHybrid(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const struct axbVec_s *xhybrid=NULL,*yhybrid=NULL;
  struct axbVec_s       *whybrid=NULL;
  struct axbScalar_s    *alpha_hybrid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = axbVecCopy(yhybrid,whybrid);CHKERRQ(ierr);
  } else {
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(yin,&yhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArrayWrite(win,&whybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecWAXPY(whybrid,alpha_hybrid,xhybrid,yhybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(yin,&yhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayWrite(win,&whybrid);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqHybrid(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode           ierr;
  PetscInt                 n = xin->map->n,j;
  struct axbVec_s          *xhybrid;
  struct axbScalar_s       **alphas_hybrid;
  const struct axbVec_s    **yin_hybrid;

  PetscFunctionBegin;
  ierr = PetscMalloc1(nv,&yin_hybrid);CHKERRQ(ierr);
  ierr = PetscMalloc1(nv,&alphas_hybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
  for (j=0; j<nv; ++j) {
    ierr = axbScalarCreate(axb_handle,&(alphas_hybrid[j]),(void*)(alpha + j),AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(y[j],&(yin_hybrid[j]));CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = axbVecMAXPY(xhybrid,nv,(const struct axbScalar_s * const *)alphas_hybrid,yin_hybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  for (j=0; j<nv; ++j) {
    ierr = VecHybridRestoreArrayRead(y[j],&(yin_hybrid[j]));CHKERRQ(ierr);
    ierr = axbScalarDestroy(alphas_hybrid[j]);
  }
  ierr = VecHybridRestoreArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
  ierr = PetscFree(yin_hybrid);CHKERRQ(ierr);
  ierr = PetscFree(alphas_hybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqHybrid(Vec xin,Vec yin,PetscScalar *z)
{
  const struct axbVec_s *xhybrid=NULL,*yhybrid=NULL;
  struct axbScalar_s    *alpha_hybrid;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)z,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecDot(xhybrid,yhybrid,alpha_hybrid);CHKERRQ(ierr);
  ierr = axbScalarGetValue(alpha_hybrid,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(yin,&yhybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecMDot_SeqHybrid(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode        ierr;
  PetscInt              i,n = xin->map->n;
  const struct axbVec_s **yin_hybrid,*xhybrid;
  struct axbScalar_s    **alpha_hybrid;

  PetscFunctionBegin;
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqHybrid not positive.");
  if (xin->map->n > 0) {
    ierr = PetscMalloc1(nv,&yin_hybrid);CHKERRQ(ierr);
    ierr = PetscMalloc1(nv,&alpha_hybrid);CHKERRQ(ierr);
    for (i=0; i<nv; ++i) {
      ierr = axbScalarCreate(axb_handle,&(alpha_hybrid[i]),(void*)(z+i),AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
      ierr = VecHybridGetArrayRead(yin[i],&yin_hybrid[i]);CHKERRQ(ierr);
    }
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = axbVecMDot(xhybrid,nv,yin_hybrid,alpha_hybrid);CHKERRQ(ierr);
    for (i=0; i<nv; ++i) {
      ierr = axbScalarGetValue(alpha_hybrid[i],(void*)(z+i),AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = axbScalarDestroy(alpha_hybrid[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(yin_hybrid);CHKERRQ(ierr);
    ierr = PetscFree(alpha_hybrid);CHKERRQ(ierr);
  } else { /* local size zero */
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }
  ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode VecSet_SeqHybrid(Vec xin,PetscScalar alpha)
{
  struct axbVec_s    *xhybrid;
  struct axbScalar_s *alpha_hybrid;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecHybridGetArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  ierr = axbVecSet(xhybrid, alpha_hybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqHybrid(Vec xin,PetscScalar alpha)
{
  struct axbVec_s    *xhybrid;
  struct axbScalar_s *alpha_hybrid;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqHybrid(xin,alpha);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecScale(xhybrid, alpha_hybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayWrite(xin,&xhybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  }
  ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqHybrid(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqHybrid(xin,yin,z);CHKERRQ(ierr); /* Note: This needs to be changed as soon as VecHybrid supports complex arithmetic */
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqHybrid(Vec xin,Vec yin)
{
  const struct axbVec_s *xhybrid;
  struct axbVec_s       *yhybrid;
  PetscErrorCode        ierr;
  PetscBool             copy_on_gpu = PETSC_FALSE;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) copy_on_gpu = PETSC_TRUE;
    else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      if (yin->offloadmask == PETSC_OFFLOAD_GPU || yin->offloadmask == PETSC_OFFLOAD_BOTH) copy_on_gpu = PETSC_TRUE;
    } 

    if (copy_on_gpu) {
      ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
      ierr = VecHybridGetArrayWrite(yin,&yhybrid);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      ierr = axbVecCopy(xhybrid,yhybrid);CHKERRQ(ierr);
      ierr = WaitForGPU();CHKERRQ(ierr);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
      ierr = VecHybridRestoreArrayWrite(yin,&yhybrid);CHKERRQ(ierr);
    } else {
      ierr = VecCopy_SeqHybrid_Private(xin,yin);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqHybrid(Vec xin,Vec yin)
{
  struct axbVec_s *xhybrid,*yhybrid;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecHybridGetArray(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecSwap(xhybrid,yhybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecHybridRestoreArray(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArray(yin,&yhybrid);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqHybrid(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscScalar           a = alpha,b = beta;
  const struct axbVec_s *xhybrid;
  struct axbVec_s       *yhybrid;
  struct axbScalar_s    *alpha_hybrid,*beta_hybrid;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_SeqHybrid(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqHybrid(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqHybrid(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecCopy(xhybrid,yhybrid);CHKERRQ(ierr);
    ierr = axbVecScale(yhybrid,alpha_hybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
  } else {
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)&alpha,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = axbScalarCreate(axb_handle,&beta_hybrid,(void*)&beta,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridGetArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbVecScale(yhybrid,beta_hybrid);CHKERRQ(ierr);
    ierr = axbVecAXPY(yhybrid,alpha_hybrid,xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArray(yin,&yhybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
    ierr = axbScalarDestroy(beta_hybrid);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRQ(ierr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqHybrid(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqHybrid(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqHybrid(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqHybrid(zin,gamma);CHKERRQ(ierr);
    ierr = VecAXPY_SeqHybrid(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqHybrid(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqHybrid(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode        ierr;
  PetscInt              n = xin->map->n;
  const struct axbVec_s *xhybrid;
  struct axbScalar_s    *alpha_hybrid;

  PetscFunctionBegin;
  if (n > 0) {
    ierr = VecHybridGetArrayRead(xin,&xhybrid);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    ierr = axbScalarCreate(axb_handle,&alpha_hybrid,(void*)z,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      ierr = axbVecNorm2(xhybrid,alpha_hybrid);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_hybrid,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) {
      ierr = axbVecNormInf(xhybrid,alpha_hybrid);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_hybrid,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
    } else if (type == NORM_1) {
      ierr = axbVecNorm1(xhybrid,alpha_hybrid);CHKERRQ(ierr);
      ierr = axbScalarGetValue(alpha_hybrid,(void*)z,AXB_REAL_DOUBLE);CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    } else if (type == NORM_1_AND_2) {
      ierr = VecNorm_SeqHybrid(xin,NORM_1,z);CHKERRQ(ierr);
      ierr = VecNorm_SeqHybrid(xin,NORM_2,z+1);CHKERRQ(ierr);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = axbScalarDestroy(alpha_hybrid);CHKERRQ(ierr);
    ierr = VecHybridRestoreArrayRead(xin,&xhybrid);CHKERRQ(ierr);
  } else {
    z[0] = 0.0;
    if (type == NORM_1_AND_2) z[1] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqHybrid(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode        ierr;
  PetscReal             n=s->map->n;
  const struct axbVec_s *shybrid,*thybrid;
  struct axbScalar_s    *st_hybrid,*norm_hybrid;

  PetscFunctionBegin;
  ierr = axbScalarCreate(axb_handle,&st_hybrid,(void*)dp,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = axbScalarCreate(axb_handle,&norm_hybrid,(void*)nm,AXB_REAL_DOUBLE,axb_mem_backend);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(s,&shybrid);CHKERRQ(ierr);
  ierr = VecHybridGetArrayRead(t,&thybrid);CHKERRQ(ierr);
  ierr = axbVecDotNorm2(shybrid,thybrid,st_hybrid,norm_hybrid);CHKERRQ(ierr);
  ierr = axbScalarGetValue(st_hybrid,(void*)dp,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = axbScalarGetValue(norm_hybrid,(void*)nm,AXB_REAL_DOUBLE);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(s,&shybrid);CHKERRQ(ierr);
  ierr = VecHybridRestoreArrayRead(t,&thybrid);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqHybrid(Vec v)
{
  PetscErrorCode ierr;
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  Vec_Hybrid     *vechybrid = (Vec_Hybrid*)v->spptr;

  PetscFunctionBegin;
  if (v->spptr) {
    if (vechybrid->vec) {
      ierr = axbVecDestroy(vechybrid->vec);CHKERRQ(ierr);
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

PetscErrorCode VecConjugate_SeqHybrid(Vec xin)
{
  PetscFunctionBegin;
  /* Add code here when adding support for complex arithmetic */
  PetscFunctionReturn(0);
}


/*@C
   VecHybridGetArray - Provides access to the hybrid buffer (a axbVec_t from libaxb) inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecHybridGetArray() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The axbVec_t has to be released by calling
   VecHybridRestoreArray().  Upon restoring the vector data
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

.seealso: VecHybridRestoreArray(), VecHybridGetArrayRead(), VecHybridGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridGetArray(Vec v, struct axbVec_s **vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  *vec   = 0;
  ierr = VecHybridCopyToGPU(v);CHKERRQ(ierr);
  *vec   = ((Vec_Hybrid*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecHybridRestoreArray - Restore a hybrid device object previously acquired with VecHybridGetArray().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecHybridRestoreArray() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHybridGetArray(), VecHybridGetArrayRead(), VecHybridGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridRestoreArray(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecHybridGetArrayRead - Provides read access to the hybrid buffer (a axbVec_t from libaxb) inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecHybridGetArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The returned pointer has to be released by calling
   VecHybridRestoreArrayRead().  If the data on the host side was
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

.seealso: VecHybridRestoreArrayRead(), VecHybridGetArray(), VecHybridGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridGetArrayRead(Vec v, const struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  *a   = 0;
  ierr = VecHybridCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_Hybrid*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecHybridRestoreArrayRead - Restore a hybrid buffer (a axbVec_t from libaxb) previously acquired with VecHybridGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecHybridRestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHybridGetArrayRead(), VecHybridGetArrayWrite(), VecHybridGetArray(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridRestoreArrayRead(Vec v, const struct axbVec_s **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  *a = NULL;
  PetscFunctionReturn(0);
}

/*@C
   VecHybridGetArrayWrite - Provides write access to the hybrid buffer (a axbVec_t from libaxb) inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The returned pointer needs to be released with
   VecHybridRestoreArrayWrite().  When the pointer is released the
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

.seealso: VecHybridRestoreArrayWrite(), VecHybridGetArray(), VecHybridGetArrayRead(), VecHybridGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridGetArrayWrite(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  *a   = 0;
  ierr = VecHybridAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_Hybrid*)v->spptr)->vec;
  PetscFunctionReturn(0);
}

/*@C
   VecHybridRestoreArrayWrite - Restore a hybrid buffer (a axbVec_t from libaxb) previously acquired with VecHybridGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the internal axbVec_t object providing hybrid capabilities.  This pointer is invalid after
       VecHybridRestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecHybridGetArrayWrite(), VecHybridGetArray(), VecHybridGetArrayRead(), VecHybridGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecHybridRestoreArrayWrite(Vec v, struct axbVec_s **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQHYBRID,VECMPIHYBRID);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

