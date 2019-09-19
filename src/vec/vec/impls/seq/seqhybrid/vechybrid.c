/*
 Implementation of the sequential cuda vectors.

 This file contains the code that can be compiled with a C
 compiler.  The companion file veccuda2.cu contains the code that
 must be compiled with nvcc or a C++ compiler.
 */

#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I <petscvec.h> I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqhybrid/vechybridimpl.h>


/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscHybridFlag for the vector
    Does NOT zero the Hybrid array
 */
PetscErrorCode VecHybridAllocateCheckHost(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec_Seq        *s = (Vec_Seq*)v->data;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  if (!s) {
    ierr = PetscNewLog((PetscObject)v,&s);CHKERRQ(ierr);
    v->data = s;
  }
  if (!s->array) {
    ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
    if (v->valid_GPU_array == PETSC_OFFLOAD_UNALLOCATED) {
      v->valid_GPU_array = PETSC_OFFLOAD_CPU;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqHybrid_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecHybridAllocateCheckHost(xin);CHKERRQ(ierr);
  ierr = VecHybridAllocateCheckHost(yin);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscArraycpy(ya,xa,xin->map->n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqHybrid(Vec vin)
{
  PetscErrorCode ierr;
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  ierr = VecHybridCopyFromGPU(vin);CHKERRQ(ierr);
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  vin->valid_GPU_array = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqHybrid(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHybridCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqHybrid(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecHybridCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/*@
 VecCreateSeqHybrid - Creates a hybrid, sequential array-style vector.

 Collective

 Input Parameter:
 +  comm - the communicator, should be PETSC_COMM_SELF
 -  n - the vector length

 Output Parameter:
 .  v - the vector

 Notes:
 Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
 same type as an existing vector.

 Level: intermediate

 .seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
 @*/
PetscErrorCode VecCreateSeqHybrid(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQHYBRID);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDuplicate_SeqHybrid(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqHybrid(PetscObjectComm((PetscObject)win),win->map->n,V);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}


PetscErrorCode VecPinToCPU_SeqHybrid(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->pinnedtocpu = pin;
  if (pin) {
    ierr = VecHybridCopyFromGPU(V);CHKERRQ(ierr);
    V->ops->dot                    = VecDot_Seq;
    V->ops->norm                   = VecNorm_Seq;
    V->ops->tdot                   = VecTDot_Seq;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->mtdot_local            = VecMTDot_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->mdot                   = VecMDot_Seq;
    V->ops->mtdot                  = VecMTDot_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dotnorm2               = NULL;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_Seq;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->duplicate              = VecDuplicate_Seq;
    V->ops->conjugate              = VecConjugate_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  } else {
    V->ops->dot                    = VecDot_SeqHybrid;
    V->ops->norm                   = VecNorm_SeqHybrid;
    V->ops->tdot                   = VecTDot_SeqHybrid;
    V->ops->scale                  = VecScale_SeqHybrid;
    V->ops->copy                   = VecCopy_SeqHybrid;
    V->ops->set                    = VecSet_SeqHybrid;
    V->ops->swap                   = VecSwap_SeqHybrid;
    V->ops->axpy                   = VecAXPY_SeqHybrid;
    V->ops->axpby                  = VecAXPBY_SeqHybrid;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqHybrid;
    V->ops->pointwisemult          = VecPointwiseMult_SeqHybrid;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqHybrid;
    V->ops->dot_local              = VecDot_SeqHybrid;
    V->ops->tdot_local             = VecTDot_SeqHybrid;
    V->ops->norm_local             = VecNorm_SeqHybrid;
    V->ops->mdot_local             = VecMDot_SeqHybrid;
    V->ops->maxpy                  = VecMAXPY_SeqHybrid;
    V->ops->mdot                   = VecMDot_SeqHybrid;
    V->ops->aypx                   = VecAYPX_SeqHybrid;
    V->ops->waxpy                  = VecWAXPY_SeqHybrid;
    V->ops->dotnorm2               = VecDotNorm2_SeqHybrid;
    V->ops->destroy                = VecDestroy_SeqHybrid;
    V->ops->duplicate              = VecDuplicate_SeqHybrid;
    V->ops->conjugate              = NULL;
    V->ops->getarraywrite          = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCreate_SeqHybrid(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(V->map);CHKERRQ(ierr);
  V->valid_GPU_array = PETSC_OFFLOAD_GPU;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQHYBRID on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQHYBRID);CHKERRQ(ierr);
  ierr = VecPinToCPU_SeqHybrid(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->pintocpu = VecPinToCPU_SeqHybrid;
  ierr = VecHybridAllocateCheck(V);CHKERRQ(ierr);
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
