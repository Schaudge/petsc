
/*
   This file contains routines for parallel ViennaCL vector operations.
 */
#include <petscconf.h>
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

PetscErrorCode VecDestroy_MPIViennaCL(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (((Vec_ViennaCL*)v->spptr)->array) {
      delete ((Vec_ViennaCL*)v->spptr)->array;
    }
    delete (Vec_ViennaCL*) v->spptr;
  } catch(std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecNorm_MPIViennaCL(Vec xin,NormType type,PetscReal *z)
{
  PetscReal      sum,work = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr  = VecNorm_SeqViennaCL(xin,NORM_2,&work);
    work *= work;
    ierr  = MPIU_Allreduce(&work,&sum,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    *z    = PetscSqrtReal(sum);
  } else if (type == NORM_1) {
    /* Find the local part */
    ierr = VecNorm_SeqViennaCL(xin,NORM_1,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    /* Find the local max */
    ierr = VecNorm_SeqViennaCL(xin,NORM_INFINITY,&work);CHKERRQ(ierr);
    /* Find the global max */
    ierr = MPIU_Allreduce(&work,z,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    PetscReal temp[2];
    ierr = VecNorm_SeqViennaCL(xin,NORM_1,temp);CHKERRQ(ierr);
    ierr = VecNorm_SeqViennaCL(xin,NORM_2,temp+1);CHKERRQ(ierr);
    temp[1] = temp[1]*temp[1];
    ierr = MPIU_Allreduce(temp,z,2,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    z[1] = PetscSqrtReal(z[1]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqViennaCL(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_MPIViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscScalar    sum,work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecTDot_SeqViennaCL(xin,yin,&work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&work,&sum,1,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  *z   = sum;
  PetscFunctionReturn(0);
}

PetscErrorCode VecMDot_MPIViennaCL(Vec xin,PetscInt nv,const Vec y[],PetscScalar *z)
{
  PetscScalar    awork[128],*work = awork;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (nv > 128) {
    ierr = PetscMalloc1(nv,&work);CHKERRQ(ierr);
  }
  ierr = VecMDot_SeqViennaCL(xin,nv,y,work);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(work,z,nv,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  if (nv > 128) {
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_MPIViennaCL(Vec s,Vec t,PetscScalar *dp,PetscScalar *nm)
{
  PetscErrorCode ierr;
  PetscScalar    work[2],sum[2];

  PetscFunctionBegin;
  ierr = VecDotNorm2_SeqViennaCL(s,t,work,work+1);CHKERRQ(ierr);
  ierr = MPIU_Allreduce((void*)&work,(void*)&sum,2,MPIU_SCALAR,MPIU_SUM,PetscObjectComm((PetscObject)s));CHKERRQ(ierr);
  *dp  = sum[0];
  *nm  = sum[1];
  PetscFunctionReturn(0);
}

/*MC
   VECMPIVIENNACL - VECMPIVIENNACL = "mpi:viennacl" - The basic parallel vector, modified to use ViennaCL

   Options Database Keys:
. -vec_type mpi:viennacl - sets the vector type to VECMPIVIENNACL during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateMPIWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateMPI()
M*/

PETSC_EXTERN PetscErrorCode VecDuplicate_MPI(Vec,Vec*);

PetscErrorCode VecDuplicate_MPIViennaCL(Vec win,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate_MPI(win,v);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECMPIVIENNACL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_MPIViennaCL(Vec vv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!vv->data) {ierr = VecSetType(vv,VECMPI);CHKERRQ(ierr);}

  try {
    vv->spptr = new Vec_ViennaCL;
  } catch(std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  /* allocation of memory is delayed until  VecViennaCLAllocateCheck() */
  ((Vec_ViennaCL*)vv->spptr)->array = NULL;
  ierr = PetscObjectChangeTypeName((PetscObject)vv,VECMPIVIENNACL);CHKERRQ(ierr);

  vv->ops->dotnorm2        = VecDotNorm2_MPIViennaCL;
  vv->ops->waxpy           = VecWAXPY_SeqViennaCL;
  vv->ops->duplicate       = VecDuplicate_MPIViennaCL;
  vv->ops->dot             = VecDot_MPIViennaCL;
  vv->ops->mdot            = VecMDot_MPIViennaCL;
  vv->ops->tdot            = VecTDot_MPIViennaCL;
  vv->ops->norm            = VecNorm_MPIViennaCL;
  vv->ops->scale           = VecScale_SeqViennaCL;
  vv->ops->copy            = VecCopy_SeqViennaCL;
  vv->ops->set             = VecSet_SeqViennaCL;
  vv->ops->swap            = VecSwap_SeqViennaCL;
  vv->ops->axpy            = VecAXPY_SeqViennaCL;
  vv->ops->axpby           = VecAXPBY_SeqViennaCL;
  vv->ops->maxpy           = VecMAXPY_SeqViennaCL;
  vv->ops->aypx            = VecAYPX_SeqViennaCL;
  vv->ops->axpbypcz        = VecAXPBYPCZ_SeqViennaCL;
  vv->ops->pointwisemult   = VecPointwiseMult_SeqViennaCL;
  vv->ops->setrandom       = VecSetRandom_Seq;
  vv->ops->dot_local       = VecDot_SeqViennaCL;
  vv->ops->tdot_local      = VecTDot_SeqViennaCL;
  vv->ops->norm_local      = VecNorm_SeqViennaCL;
  vv->ops->mdot_local      = VecMDot_SeqViennaCL;
  vv->ops->destroy         = VecDestroy_MPIViennaCL;
  vv->ops->pointwisedivide = VecPointwiseDivide_SeqViennaCL;
  vv->ops->placearray      = VecPlaceArray_SeqViennaCL;
  vv->ops->replacearray    = VecReplaceArray_SeqViennaCL;
  vv->ops->resetarray      = VecResetArray_SeqViennaCL;

  vv->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec v)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)v),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecSetType(v,VECSEQVIENNACL);CHKERRQ(ierr);
  } else {
    ierr = VecSetType(v,VECMPIVIENNACL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

