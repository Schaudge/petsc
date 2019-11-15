
/*
   This file contains routines for Parallel vector operations.
 */
#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/seq/seqhybrid/vechybridimpl.h>

PetscErrorCode VecDestroy_MPIHybrid(Vec v)
{
  PetscErrorCode ierr;
  Vec_Hybrid     *vec_hybrid = (Vec_Hybrid*)v->spptr;

  PetscFunctionBegin;
  if (v->spptr) {
    ierr = axbVecDestroy(vec_hybrid->vec);CHKERRQ(ierr);
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIHybrid(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqHybrid(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqHybrid(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqHybrid(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqHybrid(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqHybrid(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPIHybrid(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqHybrid(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPIHybrid(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqHybrid(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIHybrid(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqHybrid(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIHybrid(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqHybrid(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRQ(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

/*MC
   VECMPIHYBRID - VECMPIHYBRID = "mpihybrid" - The basic parallel vector, modified to use libaxb for hybrid compute devices

   Options Database Keys:
. -vec_type mpihybrid - sets the vector type to VECMPIHYBRID during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMPI()
M*/


PetscErrorCode VecDuplicate_MPIHybrid(Vec win,Vec *v)
{
  PetscErrorCode ierr;
  Vec_MPI        *vw,*w = (Vec_MPI*)win->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecCreate(PetscObjectComm((PetscObject)win),v);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*v)->map);CHKERRQ(ierr);

  ierr = VecCreate_MPI_Private(*v,PETSC_FALSE,w->nghost,0);CHKERRQ(ierr);
  vw   = (Vec_MPI*)(*v)->data;
  ierr = PetscMemcpy((*v)->ops,win->ops,sizeof(struct _VecOps));CHKERRQ(ierr);

  /* save local representation of the parallel vector (and scatter) if it exists */
  if (w->localrep) {
    ierr = VecGetArray(*v,&array);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,win->map->n+w->nghost,array,&vw->localrep);CHKERRQ(ierr);
    ierr = PetscMemcpy(vw->localrep->ops,w->localrep->ops,sizeof(struct _VecOps));CHKERRQ(ierr);
    ierr = VecRestoreArray(*v,&array);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)*v,(PetscObject)vw->localrep);CHKERRQ(ierr);
    vw->localupdate = w->localupdate;
    if (vw->localupdate) {
      ierr = PetscObjectReference((PetscObject)vw->localupdate);CHKERRQ(ierr);
    }
  }

  /* New vector should inherit stashing property of parent */
  (*v)->stash.donotstash   = win->stash.donotstash;
  (*v)->stash.ignorenegidx = win->stash.ignorenegidx;

  /* change type_name appropriately */
  ierr = PetscObjectChangeTypeName((PetscObject)(*v),VECMPIHYBRID);CHKERRQ(ierr);

  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*v))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*v))->qlist);CHKERRQ(ierr);
  (*v)->map->bs   = PetscAbs(win->map->bs);
  (*v)->bstash.bs = win->bstash.bs;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecPinToCPU_MPIHybrid(Vec V,PetscBool pin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->pinnedtocpu = pin;
  if (pin) {
    ierr = VecCUDACopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dotnorm2               = NULL;
    V->ops->waxpy                  = VecWAXPY_Seq;
    V->ops->dot                    = VecDot_MPI;
    V->ops->mdot                   = VecMDot_MPI;
    V->ops->tdot                   = VecTDot_MPI;
    V->ops->norm                   = VecNorm_MPI;
    V->ops->scale                  = VecScale_Seq;
    V->ops->copy                   = VecCopy_Seq;
    V->ops->set                    = VecSet_Seq;
    V->ops->swap                   = VecSwap_Seq;
    V->ops->axpy                   = VecAXPY_Seq;
    V->ops->axpby                  = VecAXPBY_Seq;
    V->ops->maxpy                  = VecMAXPY_Seq;
    V->ops->aypx                   = VecAYPX_Seq;
    V->ops->axpbypcz               = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult          = VecPointwiseMult_Seq;
    V->ops->placearray             = VecPlaceArray_Seq;
    V->ops->replacearray           = VecReplaceArray_Seq;
    V->ops->resetarray             = VecResetArray_Seq;
    V->ops->dot_local              = VecDot_Seq;
    V->ops->tdot_local             = VecTDot_Seq;
    V->ops->norm_local             = VecNorm_Seq;
    V->ops->mdot_local             = VecMDot_Seq;
    V->ops->pointwisedivide        = VecPointwiseDivide_Seq;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  } else {
    V->ops->dotnorm2               = VecDotNorm2_MPIHybrid;
    V->ops->waxpy                  = VecWAXPY_SeqHybrid;
    V->ops->duplicate              = VecDuplicate_MPIHybrid;
    V->ops->dot                    = VecDot_MPIHybrid;
    V->ops->mdot                   = VecMDot_MPIHybrid;
    V->ops->tdot                   = VecTDot_MPIHybrid;
    V->ops->norm                   = VecNorm_MPIHybrid;
    V->ops->scale                  = VecScale_SeqHybrid;
    V->ops->copy                   = VecCopy_SeqHybrid;
    V->ops->set                    = VecSet_SeqHybrid;
    V->ops->swap                   = VecSwap_SeqHybrid;
    V->ops->axpy                   = VecAXPY_SeqHybrid;
    V->ops->axpby                  = VecAXPBY_SeqHybrid;
    V->ops->maxpy                  = VecMAXPY_SeqHybrid;
    V->ops->aypx                   = VecAYPX_SeqHybrid;
    V->ops->axpbypcz               = VecAXPBYPCZ_SeqHybrid;
    V->ops->pointwisemult          = VecPointwiseMult_SeqHybrid;
    V->ops->placearray             = VecPlaceArray_SeqHybrid;
    V->ops->replacearray           = VecReplaceArray_SeqHybrid;
    V->ops->resetarray             = VecResetArray_SeqHybrid;
    V->ops->dot_local              = VecDot_SeqHybrid;
    V->ops->tdot_local             = VecTDot_SeqHybrid;
    V->ops->norm_local             = VecNorm_SeqHybrid;
    V->ops->mdot_local             = VecMDot_SeqHybrid;
    V->ops->destroy                = VecDestroy_MPIHybrid;
    V->ops->pointwisedivide        = VecPointwiseDivide_SeqHybrid;
    V->ops->getlocalvector         = NULL;
    V->ops->restorelocalvector     = NULL;
    V->ops->getlocalvectorread     = NULL;
    V->ops->restorelocalvectorread = NULL;
    V->ops->getarraywrite          = NULL;
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_MPIHybrid(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate_MPI_Private(vv,PETSC_FALSE,0,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPIHYBRID);CHKERRQ(ierr);
  ierr = VecPinToCPU_MPIHybrid(vv,PETSC_FALSE);CHKERRQ(ierr);
  vv->ops->pintocpu = VecPinToCPU_MPIHybrid;
  ierr = VecHybridAllocateCheck(vv);CHKERRQ(ierr);
  vv->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = VecSet(vv,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PETSC_EXTERN PetscErrorCode VecCreate_Hybrid(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQHYBRID);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPIHYBRID);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

