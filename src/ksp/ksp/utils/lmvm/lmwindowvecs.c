#include <petscdevice.h>
#include "lmvm.h"

PetscErrorCode LMWindowVecsCreate(Vec v, PetscInt m, LMWindowVecs *lmrv_p)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1);
  PetscValidLogicalCollectiveInt(v, m, 2);
  PetscCheck(m >= 0, PetscObjectComm((PetscObject)v), PETSC_ERR_ARG_OUTOFRANGE, "Requested window size %" PetscInt_FMT " is not >= 0", m);
  PetscInt n, N;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetSize(v, &N));
  PetscMPIInt rank;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank));
  Mat backing;
  PetscCall(VecCreateMatDense(v, n, rank == 0 ? m : 0, N, m, NULL, &backing));
  LMWindowVecs lmrv;
  PetscCall(PetscNew(&lmrv));
  *lmrv_p = lmrv;
  lmrv->m = m;
  lmrv->k = 0;
  lmrv->vecs = backing;
  PetscCall(VecDuplicate(v, &(lmrv->single)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMWindowVecsGetSingle(LMWindowVecs lmrv, PetscInt idx, Vec *single, PetscBool write, PetscBool check_idx)
{
  PetscFunctionBegin;
  PetscValidPointer(lmrv, 1);
  if (check_idx) {
    PetscValidLogicalCollectiveInt(lmrv->vecs, idx, 2);
    PetscCheck(idx >= 0 && idx < lmrv->k, PetscObjectComm((PetscObject)lmrv->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " >= number of inserted vecs %" PetscInt_FMT, idx, lmrv->k);
    PetscInt earliest = lmrv->k - lmrv->m;
    PetscCheck(idx >= earliest, PetscObjectComm((PetscObject)lmrv->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " < the earliest retained index % " PetscInt_FMT, idx, earliest);
  }
  PetscValidPointer(single, 3);
  PetscCheck(lmrv->write_placed_array == NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector %" PetscInt_FMT "is still checked out, call to LMWindowVecsRestoreNextWrite() or LMWindowVecsRestoreSingleWrite() may be missing", lmrv->placed_idx);
  PetscCheck(lmrv->read_placed_array == NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector %" PetscInt_FMT "is still checked out, must call LMWindowVecsRestoreSingleRead() first", lmrv->placed_idx);

  lmrv->placed_idx = idx;

  PetscInt loc = idx % lmrv->m;
  PetscInt lda;
  PetscMemType lmrv_memtype;
  PetscCall(MatDenseGetLDA(lmrv->vecs, &lda));
  PetscScalar *vecs_array;

  if (write) {
    PetscCall(MatDenseGetArrayWriteAndMemType(lmrv->vecs, &vecs_array, &lmrv_memtype));
    lmrv->write_placed_array = vecs_array;
  } else {
    PetscCall(MatDenseGetArrayReadAndMemType(lmrv->vecs, (const PetscScalar **) &vecs_array, &lmrv_memtype));
    lmrv->read_placed_array = vecs_array;
  }
  lmrv->memtype = lmrv_memtype;

  const PetscScalar *idx_start = &vecs_array[loc * lda];
  switch (lmrv_memtype) {
  case PETSC_MEMTYPE_HOST:
    PetscCall(VecPlaceArray(lmrv->single, idx_start));
    break;
  case PETSC_MEMTYPE_CUDA:
  case PETSC_MEMTYPE_NVSHMEM:
    PetscCall(VecCUDAPlaceArray(lmrv->single, idx_start));
    break;
  case PETSC_MEMTYPE_HIP:
    PetscCall(VecHIPPlaceArray(lmrv->single, idx_start));
    break;
  default:
    PetscUnreachable();
    break;
  }
  if (!write) PetscCall(VecLockReadPush(lmrv->single));
  *single = lmrv->single;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsGetNextWrite(LMWindowVecs lmrv, Vec *next_write)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsGetSingle(lmrv, lmrv->k, next_write, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsGetSingleRead(LMWindowVecs lmrv, PetscInt idx, Vec *single_read)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsGetSingle(lmrv, idx, single_read, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsGetSingleWrite(LMWindowVecs lmrv, PetscInt idx, Vec *single_write)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsGetSingle(lmrv, idx, single_write, PETSC_TRUE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMWindowVecsRestoreSingle(LMWindowVecs lmrv, Vec *single, PetscBool write)
{
  PetscFunctionBegin;
  PetscValidPointer(lmrv, 1);
  PetscValidPointer(single, 2);
  PetscCheck(*single == lmrv->single, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Trying to restore the wrong vector");
  if (lmrv->m > 0) {
    PetscCheck(write || lmrv->read_placed_array != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Single read vector is not checked out, missing call to LMWindowVecsGetSingleRead()");
    PetscCheck((!write) || lmrv->write_placed_array != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Single write vector is not checked out, missing call to LMWindowVecsGetSingleWrite() or LMReycledVecsGetNextWrite()");
    PetscMemType lmrv_memtype = lmrv->memtype;
    PetscScalar *vecs_array;
    if (write) {
      vecs_array = lmrv->write_placed_array;
      lmrv->write_placed_array = NULL;
    } else {
      vecs_array = (PetscScalar *) lmrv->read_placed_array;
      lmrv->read_placed_array = NULL;
      PetscCall(VecLockReadPop(lmrv->single));
    }
    switch (lmrv_memtype) {
    case PETSC_MEMTYPE_HOST:
      PetscCall(VecResetArray(lmrv->single));
      break;
    case PETSC_MEMTYPE_CUDA:
    case PETSC_MEMTYPE_NVSHMEM:
      PetscCall(VecCUDAResetArray(lmrv->single));
      break;
    case PETSC_MEMTYPE_HIP:
      PetscCall(VecHIPResetArray(lmrv->single));
      break;
    default:
      PetscUnreachable();
      break;
    }
    if (write) {
      PetscCall(MatDenseRestoreArrayWriteAndMemType(lmrv->vecs, &vecs_array));
    } else {
      PetscCall(MatDenseRestoreArrayReadAndMemType(lmrv->vecs, (const PetscScalar **) &vecs_array));
    }
  }
  *single = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsRestoreNextWrite(LMWindowVecs lmrv, Vec *single_write)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsRestoreSingle(lmrv, single_write, PETSC_TRUE));
  lmrv->k++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsRestoreSingleRead(LMWindowVecs lmrv, PetscInt loc, Vec *single_read)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsRestoreSingle(lmrv, single_read, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsRestoreSingleWrite(LMWindowVecs lmrv, PetscInt loc, Vec *single_write)
{
  PetscFunctionBegin;
  PetscCall(LMWindowVecsRestoreSingle(lmrv, single_write, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsDestroy(LMWindowVecs *lmrv_p)
{
  LMWindowVecs lmrv = *lmrv_p;
  *lmrv_p = NULL;
  if (lmrv == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(lmrv->write_placed_array == NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector %" PetscInt_FMT "is still checked out, must call LMWindowVecsRestoreSingleWrite() or LMWindowVecsRestoreNextWrite() first", lmrv->placed_idx);
  PetscCheck(lmrv->read_placed_array == NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vector %" PetscInt_FMT "is still checked out, must call LMWindowVecsRestoreSingleRead() first", lmrv->placed_idx);
  PetscCall(VecDestroy(&lmrv->single));
  PetscCall(MatDestroy(&lmrv->vecs));
  PetscCall(PetscFree(lmrv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode LMWindowVecsCopy(LMWindowVecs rv_a, LMWindowVecs rv_b) {
  PetscFunctionBegin;
  PetscValidPointer(rv_a, 1);
  PetscValidPointer(rv_b, 2);
  PetscCall(MatCopy(rv_a->vecs, rv_b->vecs, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}
