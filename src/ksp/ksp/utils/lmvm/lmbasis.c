#include <petscdevice.h>
#include "lmvm.h"

PetscErrorCode LMBasisCreate(Vec v, PetscInt m, LMBasis *basis_p)
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
  Mat     backing;
  VecType type;
  PetscCall(VecGetType(v, &type));
  PetscCall(MatCreateDenseFromVecType(PetscObjectComm((PetscObject)v), type, n, rank == 0 ? m : 0, N, m, n, NULL, &backing));
  LMBasis basis;
  PetscCall(PetscNew(&basis));
  *basis_p    = basis;
  basis->m    = m;
  basis->k    = 0;
  basis->vecs = backing;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMBasisGetVec_Internal(LMBasis basis, PetscInt idx, Vec *single, PetscMemoryAccessMode mode, PetscBool check_idx)
{
  PetscFunctionBegin;
  PetscValidPointer(basis, 1);
  if (check_idx) {
    PetscValidLogicalCollectiveInt(basis->vecs, idx, 2);
    PetscCheck(idx >= 0 && idx < basis->k, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " >= number of inserted vecs %" PetscInt_FMT, idx, basis->k);
    PetscInt earliest = basis->k - basis->m;
    PetscCheck(idx >= earliest, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for index %" PetscInt_FMT " < the earliest retained index % " PetscInt_FMT, idx, earliest);
  }
  switch (mode) {
  case PETSC_MEMORY_ACCESS_READ:
    PetscCall(MatDenseGetColumnVecRead(basis->vecs, idx % basis->m, single));
    break;
  case PETSC_MEMORY_ACCESS_WRITE:
    PetscCall(MatDenseGetColumnVecWrite(basis->vecs, idx % basis->m, single));
    break;
  case PETSC_MEMORY_ACCESS_READ_WRITE:
    PetscCall(MatDenseGetColumnVec(basis->vecs, idx % basis->m, single));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetNextVec(LMBasis basis, Vec *next)
{
  PetscFunctionBegin;
  PetscCall(LMBasisGetVec_Internal(basis, basis->k, next, PETSC_MEMORY_ACCESS_WRITE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetVec(LMBasis basis, PetscInt idx, PetscMemoryAccessMode mode, Vec *single)
{
  PetscFunctionBegin;
  PetscCall(LMBasisGetVec_Internal(basis, idx, single, mode, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreVec(LMBasis basis, PetscInt idx, PetscMemoryAccessMode mode, Vec *single)
{
  PetscFunctionBegin;
  PetscValidPointer(basis, 1);
  switch (mode) {
  case PETSC_MEMORY_ACCESS_READ:
    PetscCall(MatDenseRestoreColumnVecRead(basis->vecs, idx % basis->m, single));
    break;
  case PETSC_MEMORY_ACCESS_WRITE:
    PetscCall(MatDenseRestoreColumnVecWrite(basis->vecs, idx % basis->m, single));
    break;
  case PETSC_MEMORY_ACCESS_READ_WRITE:
    PetscCall(MatDenseRestoreColumnVec(basis->vecs, idx % basis->m, single));
    break;
  }
  *single = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreNextVec(LMBasis basis, Vec *next)
{
  PetscFunctionBegin;
  PetscCall(LMBasisRestoreVec(basis, basis->k, PETSC_MEMORY_ACCESS_WRITE, next));
  basis->k++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisDestroy(LMBasis *basis_p)
{
  PetscFunctionBegin;
  LMBasis basis = *basis_p;
  *basis_p      = NULL;
  if (basis == NULL) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCheck(basis->work_vecs_in_use == NULL, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_ARG_WRONGSTATE, "Work vecs are still checked out at destruction");
  {
    VecLink head = basis->work_vecs_available;
    while (head) {
      VecLink next = head->next;

      PetscCall(VecDestroy(&head->vec));
      PetscCall(PetscFree(head));
      head = next;
    }
  }
  PetscCheck(basis->work_rows_in_use == NULL, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_ARG_WRONGSTATE, "Work rows are still checked out at destruction");
  {
    RowLink head = basis->work_rows_available;
    while (head) {
      RowLink next = head->next;

      PetscCall(PetscFree(head->row));
      PetscCall(PetscFree(head));
      head = next;
    }
  }
  PetscCall(MatDestroy(&basis->vecs));
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetWorkVec(LMBasis basis, Vec *vec_p)
{
  VecLink link;

  PetscFunctionBegin;
  if (!basis->work_vecs_available) {
    PetscCall(PetscNew(&(basis->work_vecs_available)));
    PetscCall(MatCreateVecs(basis->vecs, NULL, &(basis->work_vecs_available->vec)));
  }
  link                       = basis->work_vecs_available;
  basis->work_vecs_available = link->next;
  link->next                 = basis->work_vecs_in_use;
  basis->work_vecs_in_use    = link;

  *vec_p    = link->vec;
  link->vec = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreWorkVec(LMBasis basis, Vec *vec_p)
{
  Vec     v    = *vec_p;
  VecLink link = NULL;

  PetscFunctionBegin;
  *vec_p = NULL;
  PetscCheck(basis->work_vecs_in_use, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_ARG_WRONGSTATE, "Trying to check in a vec that wasn't checked out");
  link                       = basis->work_vecs_in_use;
  basis->work_vecs_in_use    = link->next;
  link->next                 = basis->work_vecs_available;
  basis->work_vecs_available = link;

  PetscAssert(link->vec == NULL, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_PLIB, "Link not ready to return vector");
  link->vec = v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetWorkRow(LMBasis basis, PetscScalar **row_p)
{
  PetscInt m    = basis->m;
  PetscInt lead = m / 2;
  RowLink  link;

  PetscFunctionBegin;
  if (!basis->work_rows_available) {
    PetscCall(PetscNew(&(basis->work_rows_available)));
    PetscCall(PetscMalloc1(2 * m, &(basis->work_rows_available->row)));
  }
  link                       = basis->work_rows_available;
  basis->work_rows_available = link->next;
  link->next                 = basis->work_rows_in_use;
  basis->work_rows_in_use    = link;

  *row_p    = &(link->row[lead]);
  link->row = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisRestoreWorkRow(LMBasis basis, PetscScalar **row_p)
{
  PetscInt     m    = basis->m;
  PetscInt     lead = m / 2;
  PetscScalar *row  = &((*row_p)[-lead]);
  RowLink      link = NULL;

  PetscFunctionBegin;
  *row_p = NULL;
  PetscCheck(basis->work_rows_in_use, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_ARG_WRONGSTATE, "Trying to check in a row that wasn't checked out");
  link                       = basis->work_rows_in_use;
  basis->work_rows_in_use    = link->next;
  link->next                 = basis->work_rows_available;
  basis->work_rows_available = link;

  PetscAssert(link->row == NULL, PetscObjectComm((PetscObject)(basis->vecs)), PETSC_ERR_PLIB, "Link not ready to return vector");
  link->row = row;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisCopy(LMBasis basis_a, LMBasis basis_b)
{
  PetscFunctionBegin;
  PetscCheck(basis_a->m == basis_b->m, PetscObjectComm((PetscObject)basis_a), PETSC_ERR_ARG_SIZ, "Copy target has different number of vecs, %" PetscInt_FMT " != %" PetscInt_FMT, basis_b->m, basis_a->m);
  basis_b->k = basis_a->k;
  PetscCall(MatCopy(basis_a->vecs, basis_b->vecs, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGetRange(LMBasis basis, PetscInt *oldest, PetscInt *next)
{
  PetscFunctionBegin;
  *next   = basis->k;
  *oldest = PetscMax(0, basis->k - basis->m);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisGEMVH(PetscScalar alpha, LMBasis basis, PetscInt oldest, PetscInt next, Vec v, PetscScalar beta, PetscScalar y[])
{
  PetscInt m    = basis->m;
  PetscInt lead = m / 2;
  PetscInt lim  = next - oldest;
  PetscFunctionBegin;

  if (lim <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt basis_oldest, basis_next;
  PetscCall(LMBasisGetRange(basis, &basis_oldest, &basis_next));
  PetscCheck(oldest >= basis_oldest && next <= basis_next, PetscObjectComm((PetscObject)basis->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for vec that hasn't been computed or is no longer stored");

  PetscInt next_idx   = ((next - 1) % m) + 1;
  PetscInt oldest_idx = oldest % m;

  if (oldest_idx < next_idx) {
    // in order
    PetscCall(MatDenseColumnsGEMVHermitianTranspose_Private(NULL, alpha, basis->vecs, oldest_idx, next_idx, v, beta, &y[oldest - basis_oldest], 1, PETSC_MEMTYPE_HOST));
  } else if (lim == m) {
    // out of order, contiguous, rearrange for one call

    // basis is in recycle orders, we want to return dot in history order, so
    // we do the GEMVH at a shift location and memcpy into the correct order
    PetscInt tail_length = basis_oldest % m;
    PetscInt head_length = (basis_next - basis_oldest) - tail_length;

    PetscScalar *buf = y;
    if (tail_length <= lead) buf = &y[-tail_length]; // write with the head starting at dot[0], move the tail
    else buf = &y[m - tail_length];                  // write with the head starting at dot[m], move the head

    if (beta != 0.0) {
      if (tail_length <= lead) {
        // copy the newer values to the front of the
        PetscCall(PetscArraycpy(buf, &y[head_length], tail_length));
      } else {
        // copy the older values to the front
        PetscCall(PetscArraycpy(&buf[tail_length], y, head_length));
      }
    }

    PetscCall(MatDenseColumnsGEMVHermitianTranspose_Private(NULL, alpha, basis->vecs, 0, m, v, beta, buf, 1, PETSC_MEMTYPE_HOST));

    if (tail_length <= lead) {
      // copy the newer values to the end
      PetscCall(PetscArraycpy(&y[head_length], buf, tail_length));
    } else {
      // copy the older values to the front
      PetscCall(PetscArraycpy(y, &buf[tail_length], head_length));
    }
  } else {
    // out of order, discontiguous, make two calls
    PetscCall(MatDenseColumnsGEMVHermitianTranspose_Private(NULL, alpha, basis->vecs, 0, next_idx, v, beta, &y[oldest - basis_oldest + m - oldest_idx], 1, PETSC_MEMTYPE_HOST));
    PetscCall(MatDenseColumnsGEMVHermitianTranspose_Private(NULL, alpha, basis->vecs, oldest_idx, m, v, beta, &y[oldest - basis_oldest], 1, PETSC_MEMTYPE_HOST));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

// x must come from LMBasisGetWorkRow()
PETSC_INTERN PetscErrorCode LMBasisGEMV(PetscScalar alpha, LMBasis A, PetscInt oldest, PetscInt next, PetscScalar x[], PetscScalar beta, Vec y)
{
  PetscInt m    = A->m;
  PetscInt lead = m / 2;
  PetscInt lim  = next - oldest;
  PetscFunctionBegin;

  if (lim <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscInt basis_oldest, basis_next;
  PetscCall(LMBasisGetRange(A, &basis_oldest, &basis_next));
  PetscCheck(oldest >= basis_oldest && next <= basis_next, PetscObjectComm((PetscObject)A->vecs), PETSC_ERR_ARG_OUTOFRANGE, "Asked for vec that hasn't been computed or is no longer stored");

  PetscInt next_idx   = ((next - 1) % A->m) + 1;
  PetscInt oldest_idx = oldest % A->m;

  if (oldest_idx < next_idx) {
    // in order
    PetscCall(MatDenseColumnsGEMV_Private(NULL, alpha, A->vecs, oldest_idx, next_idx, &x[oldest - basis_oldest], 1, PETSC_MEMTYPE_HOST, beta, y));
  } else if (lim == A->m) {
    // out of order, contiguous, rearrange for one call

    // x is in recycle orders, we want to make it compatible with A, so
    // we have to return it to recycle order
    PetscInt tail_length = basis_oldest % m;
    PetscInt head_length = (basis_next - basis_oldest) - tail_length;

    PetscScalar *buf = x;
    if (tail_length <= lead) buf = &x[-tail_length]; // read with the head starting at dot[0]
    else buf = &x[m - tail_length];                  // read with the head starttin at dot[m]

    if (tail_length <= lead) {
      // copy the newer values to the front of the
      PetscCall(PetscArraycpy(buf, &x[head_length], tail_length));
    } else {
      // copy the older values to the front
      PetscCall(PetscArraycpy(&buf[tail_length], x, head_length));
    }
    PetscCall(MatDenseColumnsGEMV_Private(NULL, alpha, A->vecs, 0, m, buf, 1, PETSC_MEMTYPE_HOST, beta, y));
  } else {
    // out of order, discontiguous, make two calls
    PetscCall(MatDenseColumnsGEMV_Private(NULL, alpha, A->vecs, 0, next_idx, &x[oldest - basis_oldest + m - oldest_idx], 1, PETSC_MEMTYPE_HOST, beta, y));
    PetscCall(MatDenseColumnsGEMV_Private(NULL, alpha, A->vecs, oldest_idx, m, &x[oldest - basis_oldest], 1, PETSC_MEMTYPE_HOST, 1.0, y));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMBasisReset(LMBasis basis)
{
  PetscFunctionBegin;
  if (basis) basis->k = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
