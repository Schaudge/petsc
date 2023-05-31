
#include <../src/vec/vec/impls/dvecimpl.h>

typedef struct {
  VECHEADER
} Vec_Redundant;

// 
static struct _VecOps RvOps = {PetscDesignatedInitializer(duplicate, NULL), /* 1 */
                               PetscDesignatedInitializer(duplicatevecs, VecDuplicateVecs_Default),
                               PetscDesignatedInitializer(destroyvecs, VecDestroyVecs_Default),
                               PetscDesignatedInitializer(dot, VecDot_Seq),
                               PetscDesignatedInitializer(mdot, VecMDot_Seq),
                               PetscDesignatedInitializer(norm, VecNorm_Seq),
                               PetscDesignatedInitializer(tdot, VecTDot_Seq),
                               PetscDesignatedInitializer(mtdot, VecMTDot_Seq),
                               PetscDesignatedInitializer(scale, VecScale_Seq),
                               PetscDesignatedInitializer(copy, VecCopy_Seq), /* 10 */
                               PetscDesignatedInitializer(set, VecSet_Seq),
                               PetscDesignatedInitializer(swap, VecSwap_Seq),
                               PetscDesignatedInitializer(axpy, VecAXPY_Seq),
                               PetscDesignatedInitializer(axpby, VecAXPBY_Seq),
                               PetscDesignatedInitializer(maxpy, VecMAXPY_Seq),
                               PetscDesignatedInitializer(aypx, VecAYPX_Seq),
                               PetscDesignatedInitializer(waxpy, VecWAXPY_Seq),
                               PetscDesignatedInitializer(axpbypcz, VecAXPBYPCZ_Seq),
                               PetscDesignatedInitializer(pointwisemult, VecPointwiseMult_Seq),
                               PetscDesignatedInitializer(pointwisedivide, VecPointwiseDivide_Seq),
                               PetscDesignatedInitializer(setvalues, NULL), /* 20 */
                               PetscDesignatedInitializer(assemblybegin, NULL),
                               PetscDesignatedInitializer(assemblyend, NULL),
                               PetscDesignatedInitializer(getarray, NULL),
                               PetscDesignatedInitializer(getsize, NULL),
                               PetscDesignatedInitializer(getlocalsize, NULL),
                               PetscDesignatedInitializer(restorearray, NULL),
                               PetscDesignatedInitializer(max, VecMax_Seq),
                               PetscDesignatedInitializer(min, VecMin_Seq),
                               PetscDesignatedInitializer(setrandom, NULL),
                               PetscDesignatedInitializer(setoption, NULL),
                               PetscDesignatedInitializer(setvaluesblocked, NULL),
                               PetscDesignatedInitializer(destroy, NULL),
                               PetscDesignatedInitializer(view, NULL),
                               PetscDesignatedInitializer(placearray, NULL),
                               PetscDesignatedInitializer(replacearray, NULL),
                               PetscDesignatedInitializer(dot_local, VecDot_Seq),
                               PetscDesignatedInitializer(tdot_local, VecTDot_Seq),
                               PetscDesignatedInitializer(norm_local, VecNorm_Seq),
                               PetscDesignatedInitializer(mdot_local, VecMDot_Seq),
                               PetscDesignatedInitializer(mtdot_local, VecMTDot_Seq),
                               PetscDesignatedInitializer(load, NULL),
                               PetscDesignatedInitializer(reciprocal, VecReciprocal_Default),
                               PetscDesignatedInitializer(conjugate, VecConjugate_Seq),
                               PetscDesignatedInitializer(setlocaltoglobalmapping, NULL),
                               PetscDesignatedInitializer(setvalueslocal, NULL),
                               PetscDesignatedInitializer(resetarray, NULL),
                               PetscDesignatedInitializer(setfromoptions, NULL),
                               PetscDesignatedInitializer(maxpointwisedivide, VecMaxPointwiseDivide_Seq),
                               PetscDesignatedInitializer(pointwisemax, VecPointwiseMax_Seq),
                               PetscDesignatedInitializer(pointwisemaxabs, VecPointwiseMaxAbs_Seq),
                               PetscDesignatedInitializer(pointwisemin, VecPointwiseMin_Seq),
                               PetscDesignatedInitializer(getvalues, NULL),
                               PetscDesignatedInitializer(sqrt, NULL),
                               PetscDesignatedInitializer(abs, NULL),
                               PetscDesignatedInitializer(exp, NULL),
                               PetscDesignatedInitializer(log, NULL),
                               PetscDesignatedInitializer(shift, NULL),
                               PetscDesignatedInitializer(create, NULL),
                               PetscDesignatedInitializer(stridegather, VecStrideGather_Default),
                               PetscDesignatedInitializer(stridescatter, VecStrideScatter_Default),
                               PetscDesignatedInitializer(dotnorm2, NULL),
                               PetscDesignatedInitializer(getsubvector, NULL),
                               PetscDesignatedInitializer(restoresubvector, NULL),
                               PetscDesignatedInitializer(getarrayread, NULL),
                               PetscDesignatedInitializer(restorearrayread, NULL),
                               PetscDesignatedInitializer(stridesubsetgather, VecStrideSubSetGather_Default),
                               PetscDesignatedInitializer(stridesubsetscatter, VecStrideSubSetScatter_Default),
                               PetscDesignatedInitializer(viewnative, NULL),
                               PetscDesignatedInitializer(loadnative, NULL),
                               PetscDesignatedInitializer(createlocalvector, NULL),
                               PetscDesignatedInitializer(getlocalvector, NULL),
                               PetscDesignatedInitializer(restorelocalvector, NULL),
                               PetscDesignatedInitializer(getlocalvectorread, NULL),
                               PetscDesignatedInitializer(restorelocalvectorread, NULL),
                               PetscDesignatedInitializer(bindtocpu, NULL),
                               PetscDesignatedInitializer(getarraywrite, NULL),
                               PetscDesignatedInitializer(restorearraywrite, NULL),
                               PetscDesignatedInitializer(getarrayandmemtype, NULL),
                               PetscDesignatedInitializer(restorearrayandmemtype, NULL),
                               PetscDesignatedInitializer(getarrayreadandmemtype, NULL),
                               PetscDesignatedInitializer(restorearrayreadandmemtype, NULL),
                               PetscDesignatedInitializer(getarraywriteandmemtype, NULL),
                               PetscDesignatedInitializer(restorearraywriteandmemtype, NULL),
                               PetscDesignatedInitializer(concatenate, NULL),
                               PetscDesignatedInitializer(sum, NULL),
                               PetscDesignatedInitializer(setpreallocationcoo, NULL),
                               PetscDesignatedInitializer(setvaluescoo, NULL),
                               PetscDesignatedInitializer(errorwnorm, NULL)};

/*
  VecCreate_Redundant

*/
PetscErrorCode VecCreate_Redundant_Private(Vec v, PetscBool alloc, const PetscScalar array[])
{
  PetscInt       size, rank;
  MPI_Comm       comm;
  Vec_Redundant *s;

  PetscFunctionBegin;
  PetscCall(PetscNew(&s));
  v->data = (void *)s;
  PetscCall(PetscMemcpy(v->ops, &RvOps, sizeof(RvOps)));
  v->petscnative = PETSC_TRUE;

  comm = PetscObjectComm((PetscObject) v);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (array) v->offloadmask = PETSC_OFFLOAD_CPU;

  PetscCheck(v->map->N >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must specify global size N >= 0 for a redundant vector, cannot be PETSC_DECIDE or PETSC_DETERMINE");
  PetscCheck(rank == size - 1 || v->map->n == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "By convention the local sizes of all MPI processes but the last are 0 for a redundant vector");
  PetscCall(PetscLayoutSetUp(v->map));

  s->array           = (PetscScalar *)array;
  s->array_allocated = NULL;
  if (alloc && !array) {
    PetscInt N = v->map->N;
    PetscCall(PetscCalloc1(N, &s->array));
    s->array_allocated = s->array;
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_2], 0));
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_1], 0));
    PetscCall(PetscObjectComposedDataSetReal((PetscObject)v, NormIds[NORM_INFINITY], 0));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)v, VECMPI));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_USE_DEBUG)

  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
      PetscInt b1[2], b2[2]; \
      b1[0] = -b; \
      b1[1] = b; \
      PetscCall(MPIU_Allreduce(b1, b2, 2, MPIU_INT, MPI_MAX, a)); \
      PetscCheck(-b2[0] == b2[1], a, PETSC_ERR_ARG_WRONG, "Int value must be same on all processes, argument # %d", c); \
    } while (0)
#else
  #define PetscValidLogicalCollectiveIntComm(a, b, c) \
    do { \
      (void)(a); \
      (void)(b); \
    } while (0)
#endif

PetscErrorCode VecCreateRedundant(MPI_Comm comm, PetscInt N, Vec *v)
{
  PetscInt       size, rank;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveIntComm(comm, N, 2);
  PetscCheck(N >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Must specify global size N >= 0 for a redundant vector, cannot be PETSC_DECIDE or PETSC_DETERMINE");
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(VecCreate(comm, v));
  PetscCall(VecSetSizes(*v, rank == size - 1 ? N : 0, N));
  PetscCall(VecSetType(*v, VECREDUNDANT));
  PetscFunctionReturn(PETSC_SUCCESS);
}
