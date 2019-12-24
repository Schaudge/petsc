
/*
   Implements the sequential vectors.
*/

#include <../src/vec/vec/impls/dvecimpl.h>          /*I "petscvec.h" I*/
/*MC
   VECSEQ - VECSEQ = "seq" - The basic sequential vector

   Options Database Keys:
. -vec_type seq - sets the vector type to VECSEQ during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

PETSC_EXTERN PetscErrorCode VecCreate_Seq(Vec V)
{
  Vec_Seq        *s;
  PetscScalar    *array;
  PetscErrorCode ierr;
  PetscInt       n = PetscMax(V->map->n,V->map->N);
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  ierr = PetscNewLog(V,&s);CHKERRQ(ierr);
  ierr = PetscMemcpy(V->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);

  V->data            = (void*)s;
  V->petscnative     = PETSC_TRUE;

  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = PetscObjectComposeFunction((PetscObject)V,"PetscMatlabEnginePut_C",VecMatlabEnginePut_Default);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)V,"PetscMatlabEngineGet_C",VecMatlabEngineGet_Default);CHKERRQ(ierr);
#endif

  ierr = PetscMalloc1(n,&array);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)V, n*sizeof(PetscScalar));CHKERRQ(ierr);
  s->array           = (PetscScalar*)array;
  s->array_allocated = array;

  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
