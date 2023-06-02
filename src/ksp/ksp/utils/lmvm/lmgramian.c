#include <petscmat.h>
#include <petscblaslapack.h>
#include <petscdevice.h>
#include "lmvm.h"

PETSC_INTERN PetscErrorCode LMGramianCreate(PetscInt m, LMGramian *dots)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(dots));
  (*dots)->m     = m;
  (*dots)->lda   = -1;
  (*dots)->lwork = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianDestroy(LMGramian *dots_p)
{
  PetscFunctionBegin;
  LMGramian dots = *dots_p;
  if (dots == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  if (dots->full) {
    PetscInt lead = dots->m / 2;

    PetscScalar *dots_alloc = &dots->full[-lead - lead * dots->lda];
    PetscCall(PetscDeviceFree(NULL, dots_alloc));
    dots->full = NULL;
  }
  PetscCall(PetscDeviceFree(NULL, dots->diagonal));
  PetscCall(PetscDeviceFree(NULL, dots->diagonal_copy));
  PetscCall(PetscFree(dots->factorization));
  PetscCall(PetscFree(dots->pivots));
  PetscCall(PetscFree(dots->work));
  PetscCall(PetscFree(dots));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Copy column major 2D arrays
static PetscErrorCode PetscScalarArraycpy2D(PetscScalar dest[], PetscInt dest_ld, const PetscScalar src[], PetscInt src_ld, size_t m, size_t n)
{
  PetscFunctionBegin;
  if (m == dest_ld && m == src_ld) {
    PetscCall(PetscArraycpy(dest, src, m * n));
  } else {
    if (m == 1) {
      for (size_t i = 0; i < n; i++) dest[i * dest_ld] = src[i * src_ld];
    } else {
      for (size_t i = 0; i < n; i++) { PetscCall(PetscArraycpy(&dest[i * dest_ld], &src[i * src_ld], m)); }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   0   lo  hi  m           0   rem st. m
   +---+---+---+ 0         +---+---+---+ 0
   |   |AAA|   |           |   |   |EEE|
   +---+---+---+ lo        +---+---+---+ rem (= m - hi)
   |BBB|CCC|DDD|     =>    |   |   |AAA|
   +---+---+---+ hi        +---+---+---+ start (= rem + lo)
   |   |EEE|   |           |DDD|BBB|CCC|
   +---+---+---+ m         +---+---+---+ m  
   recycle order           history order (dots)
   */
static PetscErrorCode LMGramianComputeBlockInner(Mat X, Mat Y, PetscInt lo, PetscInt hi, PetscInt m, PetscScalar dots[], PetscInt lda, LMBlockType block_type)
{
  PetscInt rem   = m - hi;
  PetscInt start = rem + lo;

  PetscFunctionBegin;
  switch (block_type) {
  case LMBLOCK_ALL:
    // compute AAA and EEE in their final positions

    // AAA
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, lo, Y, lo, hi, 0.0, &dots[rem + start * lda], lda, PETSC_MEMTYPE_HOST));
    // EEE
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, hi, m, Y, lo, hi, 0.0, &dots[0 + start * lda], lda, PETSC_MEMTYPE_HOST));
  case LMBLOCK_LOWER_TRIANGLE: // ^^^ intentional fall through, no break above
    if (rem <= hi) {
      // Compute BBB , CCC , DDD with DDD hanging off the right
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, lo, hi, Y, 0, m, 0.0, &dots[start + rem * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy DDD back to the front
      PetscCall(PetscScalarArraycpy2D(&dots[start + 0 * lda], lda, &dots[start + m * lda], lda, hi - lo, rem));
    } else {
      // Compute BBB , CCC , DDD with BBB , CCC hanging off the left
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, lo, hi, Y, 0, m, 0.0, &dots[start + -hi * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy BBB, CCC back to the end
      PetscCall(PetscScalarArraycpy2D(&dots[start + rem * lda], lda, &dots[start + -hi * lda], lda, hi - lo, hi));
    }
    break;
  case LMBLOCK_UPPER_TRIANGLE:
    if (rem <= hi) {
      // Compute AAA ; CCC ; EEE with EEE hanging off the bottom
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, m, Y, lo, hi, 0.0, &dots[rem + start * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy EEE back to the top
      PetscCall(PetscScalarArraycpy2D(&dots[0 + start * lda], lda, &dots[m + start * lda], lda, rem, hi - lo));
    } else {
      // Compute AAA ; CCC ; EEE with AAA ; CCC hanging off the top
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, m, Y, lo, hi, 0.0, &dots[-hi + start * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy AAA ; CCC back to the top
      PetscCall(PetscScalarArraycpy2D(&dots[rem + start * lda], lda, &dots[-hi + start * lda], lda, hi, hi - lo));
    }
    break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   0   lo  hi  m           0   rem st. m
   +---+---+---+ 0         +---+---+---+ 0
   |AAA|BBB|CCC|           |   |EEE|DDD|
   +---+---+---+ lo        +---+---+---+ rem (= hi - lo)
   |DDD|   |EEE|     =>    |GGG|HHH|FFF|
   +---+---+---+ hi        +---+---+---+ start (= rem + (m-hi))
   |FFF|GGG|HHH|           |BBB|CCC|AAA|
   +---+---+---+ m         +---+---+---+ end
   recycle order           history order (dots)
   */
static PetscErrorCode LMGramianComputeBlockOuter(Mat X, Mat Y, PetscInt lo, PetscInt hi, PetscInt m, PetscScalar dots[], PetscInt lda, LMBlockType block_type)
{
  PetscInt rem   = hi - lo;
  PetscInt start = rem + (m - hi);

  PetscFunctionBegin;
  switch (block_type) {
  case LMBLOCK_ALL:
    // compute EEE and DDD ; FFF in their final positions

    // EEE
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, lo, hi, Y, hi, m, 0.0, &dots[0 + rem * lda], lda, PETSC_MEMTYPE_HOST));
    // DDD ; FFF
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, lo, m, Y, 0, lo, 0.0, &dots[0 + start * lda], lda, PETSC_MEMTYPE_HOST));
  case LMBLOCK_LOWER_TRIANGLE: // ^^^ intentional fall through, no break above
    // Compute GGG , HHH in their final positions
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, hi, m, Y, lo, m, 0.0, &dots[rem + 0 * lda], lda, PETSC_MEMTYPE_HOST));
    if (lo <= m - lo) {
      // Compute AAA , BBB , CCC with AAA hanging off the left
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, lo, Y, 0, m, 0.0, &dots[start - lo * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy AAA back to the end
      PetscCall(PetscScalarArraycpy2D(&dots[start + start * lda], lda, &dots[start + -lo * lda], lda, lo, lo));
    } else {
      // Compute AAA , BBB , CCC with BBB , CCC hanging off the right
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, lo, Y, 0, m, 0.0, &dots[start + start * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy BBB , CCC back to the front
      PetscCall(PetscScalarArraycpy2D(&dots[start + 0 * lda], lda, &dots[start + m * lda], lda, lo, m - lo));
    }
    break;
  case LMBLOCK_UPPER_TRIANGLE:
    // Compute EEE ; HHH in their final positions
    PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, lo, m, Y, hi, m, 0.0, &dots[0 + rem * lda], lda, PETSC_MEMTYPE_HOST));
    if (lo <= m - lo) {
      // Compute AAA , DDD , FFF with AAA hanging off the top
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, m, Y, 0, lo, 0.0, &dots[-lo + start * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy AAA back to the end
      PetscCall(PetscScalarArraycpy2D(&dots[start + start * lda], lda, &dots[-lo + start * lda], lda, lo, lo));
    } else {
      // Compute AAA , DDD , FFF with DDD ; FFF hanging off the bottom
      PetscCall(MatDenseColumnsGEMMHermitianTranspose_Private(NULL, 1.0, X, 0, m, Y, 0, lo, 0.0, &dots[start + start * lda], lda, PETSC_MEMTYPE_HOST));
      // Copy DDD ; FFF back to the tup
      PetscCall(PetscScalarArraycpy2D(&dots[0 + start * lda], lda, &dots[m + start * lda], lda, m - lo, lo));
    }
    break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMGramianAllocateDiagonal(LMGramian dots)
{
  PetscFunctionBegin;
  if (!dots->diagonal) PetscCall(PetscDeviceCalloc(NULL, PETSC_MEMTYPE_HOST, dots->m, &dots->diagonal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMGramianAllocateFull(LMGramian dots)
{
  PetscFunctionBegin;
  if (!dots->full) {
    PetscScalar *dots_alloc;
    PetscInt     lead = dots->m / 2;

    dots->lda = PetscMax(2 * dots->m, dots->lda);
    PetscCall(PetscDeviceCalloc(NULL, PETSC_MEMTYPE_HOST, (2 * dots->m + lead) * dots->lda, &dots_alloc));
    dots->full = &dots_alloc[lead + lead * dots->lda];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianUpdateNextIndex(LMGramian dots, PetscInt k)
{
  PetscInt oldest      = PetscMax(0, dots->k - dots->m);
  PetscInt new_oldest  = PetscMax(0, k - dots->m);
  PetscInt oldest_diff = new_oldest - oldest;

  PetscFunctionBegin;
  PetscCheck(k >= dots->k, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Trying to move k backwards");
  PetscAssert(oldest_diff >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trying to move oldest backwards");
  dots->k = k;
  if (oldest_diff == 0) PetscFunctionReturn(PETSC_SUCCESS);
  if (dots->diagonal) {
    PetscInt next        = dots->status[LMBLOCK_DIAGONAL].next;
    PetscInt copy_length = next - new_oldest;
    if (copy_length > 0) {
      PetscAssert(copy_length <= dots->m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Trying to copy too many");
      // prepare the diagonal
      // shift back, overwriting the entries that are no longer needed
      if (!dots->diagonal_copy) PetscCall(PetscDeviceMalloc(NULL, PETSC_MEMTYPE_HOST, dots->m, &dots->diagonal_copy));
      PetscCall(PetscArraycpy(dots->diagonal_copy, &(dots->diagonal[oldest_diff]), copy_length));
      PetscCall(PetscArraycpy(dots->diagonal, dots->diagonal_copy, copy_length));
    }
  }
  if (dots->full) {
    PetscInt copy_length;
    PetscInt next = dots->status[LMBLOCK_ALL].next;

    next        = PetscMax(next, dots->status[LMBLOCK_LOWER_TRIANGLE].next);
    next        = PetscMax(next, dots->status[LMBLOCK_UPPER_TRIANGLE].next);
    copy_length = next - new_oldest;
    if (copy_length > 0) {
      // prepare the array
      // shift back, overwriting the entries that are no longer needed
      PetscCall(PetscArraycpy(&dots->full[dots->m * dots->lda], &(dots->full[oldest_diff + oldest_diff * dots->lda]), copy_length + (copy_length - 1) * (dots->lda)));
      PetscCall(PetscArraycpy(dots->full, &dots->full[dots->m * dots->lda], copy_length + (copy_length - 1) * (dots->lda)));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianPrepareBlock(LMGramian dots, LMBasis X, LMBasis Y, LMBlockType block_type)
{
  PetscInt      oldest, next;
  PetscObjectId operator_id    = (X->operator_id == 0) ? Y->operator_id : X->operator_id;
  PetscObjectId operator_state = (X->operator_id == 0) ? Y->operator_state : X->operator_state;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  if (dots->status[block_type].operator_id != operator_id || dots->status[block_type].operator_state != operator_state) {
    // invalidate the block
    dots->status[block_type].operator_id    = operator_id;
    dots->status[block_type].operator_state = operator_state;
    dots->status[block_type].next           = oldest;
  }
  dots->status[block_type].next = PetscMax(oldest, dots->status[block_type].next);

  // allocate
  if (block_type == LMBLOCK_DIAGONAL) PetscCall(LMGramianAllocateDiagonal(dots));
  else PetscCall(LMGramianAllocateFull(dots));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMGramianUpdateBlock_Internal(LMGramian dots, LMBasis X, LMBasis Y, LMBlockType block_type, PetscInt oldest, PetscInt next, PetscBool force)
{
  PetscInt x_oldest, x_next;
  MPI_Comm comm = PetscObjectComm((PetscObject)X->vecs);

  PetscFunctionBegin;
  PetscAssert(X->m == Y->m && X->m == dots->m, comm, PETSC_ERR_ARG_INCOMP, "X vecs, Y vecs, and dots incompatible in size, (%d, %d, %d)", (int)X->m, (int)Y->m, (int)dots->m);
  PetscAssert(X->k == Y->k, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are incompatible in state, (%d, %d)", (int)X->k, (int)Y->k);
  PetscAssert(dots->k <= X->k, comm, PETSC_ERR_ARG_INCOMP, "Dot products are ahead of X and Y, (%d, %d)", (int)dots->k, (int)X->k);
  PetscAssert(X->operator_id == 0 || Y->operator_id == 0 || X->operator_id == Y->operator_id, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are from different operators");
  PetscAssert(X->operator_id != Y->operator_id || Y->operator_state == X->operator_state, comm, PETSC_ERR_ARG_INCOMP, "X and Y vecs are from different operator states");

  PetscCall(LMGramianPrepareBlock(dots, X, Y, block_type));
  PetscCall(LMGramianUpdateNextIndex(dots, X->k));

  PetscInt start = force ? oldest : dots->status[block_type].next;
  PetscCall(LMBasisGetRange(X, &x_oldest, &x_next));
  if (start < next) {
    if (block_type == LMBLOCK_DIAGONAL) {
      for (PetscInt i = start; i < next; i++) {
        Vec x, y;
        PetscCall(LMBasisGetVec(X, i, PETSC_MEMORY_ACCESS_READ, &x));
        y = x;
        if (Y != X) PetscCall(LMBasisGetVec(Y, i, PETSC_MEMORY_ACCESS_READ, &y));
        PetscCall(VecDot(y, x, &(dots->diagonal[i - x_oldest])));
        if (Y != X) PetscCall(LMBasisRestoreVec(Y, i, PETSC_MEMORY_ACCESS_READ, &y));
        PetscCall(LMBasisRestoreVec(X, i, PETSC_MEMORY_ACCESS_READ, &x));
      }
      if (!force) dots->status[block_type].next = next;
    } else {
      PetscInt next_idx  = ((next - 1) % dots->m) + 1;
      PetscInt start_idx = start % dots->m;
      if (next_idx > start_idx) {
        PetscCall(LMGramianComputeBlockInner(X->vecs, Y->vecs, start_idx, next_idx, PetscMin(dots->m, next), dots->full, dots->lda, block_type));
      } else {
        PetscCall(LMGramianComputeBlockOuter(X->vecs, Y->vecs, next_idx, start_idx, dots->m, dots->full, dots->lda, block_type));
      }

      // each of L, U, and ALL will also update D, so copy out the diagonal
      PetscCall(LMGramianPrepareBlock(dots, X, Y, LMBLOCK_DIAGONAL));
      PetscInt diag_start = force ? oldest : dots->status[LMBLOCK_DIAGONAL].next;
      for (PetscInt i = diag_start; i < next; i++) {
        PetscInt idx        = (i - x_oldest);
        dots->diagonal[idx] = dots->full[idx + idx * dots->lda];
      }

      if (!force) {
        dots->status[block_type].next       = next;
        dots->status[LMBLOCK_DIAGONAL].next = next;

        // ALL updates L and U as well
        if (block_type == LMBLOCK_ALL) {
          dots->status[LMBLOCK_LOWER_TRIANGLE] = dots->status[LMBLOCK_ALL];
          dots->status[LMBLOCK_UPPER_TRIANGLE] = dots->status[LMBLOCK_ALL];
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// dots = X^H Y
PETSC_INTERN PetscErrorCode LMGramianUpdateBlock(LMGramian dots, LMBasis X, LMBasis Y, LMBlockType block_type)
{
  PetscInt oldest, next;

  PetscFunctionBegin;
  PetscCall(LMBasisGetRange(X, &oldest, &next));
  PetscCall(LMGramianUpdateBlock_Internal(dots, X, Y, block_type, oldest, next, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianForceUpdateBlock(LMGramian dots, LMBasis X, LMBasis Y, LMBlockType block_type, PetscInt oldest, PetscInt next)
{
  PetscFunctionBegin;
  PetscCall(LMGramianUpdateBlock_Internal(dots, X, Y, block_type, oldest, next, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// does not update
PETSC_INTERN PetscErrorCode LMGramianBlockRead(LMGramian dots, LMBlockType block_type, const PetscScalar *block[], PetscInt *lda)
{
  PetscFunctionBegin;
  if (block_type == LMBLOCK_DIAGONAL) {
    *block = dots->diagonal;
    *lda   = 1;
  } else {
    *block = dots->full;
    *lda   = dots->lda;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// dots = X^H Y
PETSC_INTERN PetscErrorCode LMGramianGetBlock(LMGramian dots, LMBasis X, LMBasis Y, LMBlockType block_type, const PetscScalar *block[], PetscInt *lda)
{
  PetscFunctionBegin;
  PetscCall(LMGramianUpdateBlock(dots, X, Y, block_type));
  PetscCall(LMGramianBlockRead(dots, block_type, block, lda));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianCopy(LMGramian src, LMGramian dest)
{
  PetscFunctionBegin;
  PetscCheck(dest->m == src->m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot copy to LMGramian of different size");
  dest->k = src->k;
  if (src->full) {
    PetscCall(LMGramianAllocateFull(dest));

    for (PetscInt i = 0; i < src->m; i++) { PetscCall(PetscArraycpy(&dest->full[i * dest->lda], &src->full[i * src->lda], src->m)); }
    dest->status[LMBLOCK_LOWER_TRIANGLE] = src->status[LMBLOCK_LOWER_TRIANGLE];
    dest->status[LMBLOCK_UPPER_TRIANGLE] = src->status[LMBLOCK_UPPER_TRIANGLE];
    dest->status[LMBLOCK_ALL]            = src->status[LMBLOCK_ALL];
  }
  if (src->diagonal) {
    PetscCall(LMGramianAllocateDiagonal(dest));
    PetscCall(PetscArraycpy(dest->diagonal, src->diagonal, src->m));
    dest->status[LMBLOCK_DIAGONAL] = src->status[LMBLOCK_DIAGONAL];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianAXPY(LMGramian Y, LMBlockType y_block_type, PetscScalar alpha, LMGramian X, LMBlockType x_block_type)
{
  PetscInt m;

  PetscFunctionBegin;
  PetscCheck(X->m == Y->m && X->k == Y->k, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot copy to LMGramian of different size");
  m = X->m;
  if (y_block_type == LMBLOCK_DIAGONAL) {
    PetscCheck(x_block_type == y_block_type, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot add nondiagonal onto diagonal");
    for (PetscInt i = 0; i < m; i++) Y->diagonal[i] += alpha * X->diagonal[i];
  } else {
    if (x_block_type == LMBLOCK_DIAGONAL) {
      PetscInt stride = X->lda + 1;
      for (PetscInt i = 0; i < m; i++) Y->full[i * stride] += X->diagonal[i];
    } else {
      PetscInt           src_stride     = X->lda;
      PetscInt           dst_stride     = Y->lda;
      PetscInt           initial_length = m;
      PetscInt           length_change  = 0;
      PetscScalar       *_dst           = Y->full;
      const PetscScalar *_src           = X->full;

      switch (x_block_type) {
      case LMBLOCK_LOWER_TRIANGLE:
        src_stride    = X->lda + 1;
        dst_stride    = Y->lda + 1;
        length_change = -1;
        break;
      case LMBLOCK_STRICT_LOWER_TRIANGLE:
        _src += 1;
        _dst += 1;
        initial_length = m - 1;
        src_stride     = X->lda + 1;
        dst_stride     = Y->lda + 1;
        length_change  = -1;
        break;
      case LMBLOCK_UPPER_TRIANGLE:
        initial_length = 1;
        length_change  = 1;
        break;
      case LMBLOCK_STRICT_UPPER_TRIANGLE:
        initial_length = 0;
        length_change  = 1;
      default:
        break;
      }
      for (PetscInt i = 0, length = initial_length; i < m; i++, _src += src_stride, _dst += dst_stride, length += length_change) {
        for (PetscInt j = 0; j < length; j++) _dst[j] += alpha * _src[j];
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LMGramianComputeFactorization(LMGramian dots, LMSolveType solve_type)
{
  PetscFunctionBegin;
  LMBlockType    block_type = LMBlockTypeFromSolveType(solve_type);
  LMBlockStatus *b_status   = &dots->status[block_type];
  LMBlockStatus *f_status   = &dots->status[LMBLOCK_FACTORIZATION];
  if (f_status->next == b_status->next && f_status->operator_id == b_status->operator_id && f_status->operator_state == b_status->operator_state) { PetscFunctionReturn(PETSC_SUCCESS); }
  *f_status = *b_status;
  if (!dots->factorization) PetscCall(PetscMalloc1(dots->m * dots->lda, &dots->factorization));
  PetscCall(PetscArraycpy(dots->factorization, dots->full, dots->m * dots->lda));
  if (!dots->pivots) PetscCall(PetscMalloc1(dots->m, &dots->pivots));
  switch (solve_type) {
  case LMSOLVE_LU: {
    PetscInt     n = PetscMin(dots->k, dots->m);
    PetscBLASInt _n;
    PetscBLASInt _lda;
    PetscBLASInt _info;
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(dots->lda, &_lda));
    PetscCallBLAS("LAPACKgetrf", LAPACKgetrf_(&_n, &_n, dots->factorization, &_lda, dots->pivots, &_info));
    PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XGETRF INFO=%d", (int)_info);
    break;
  }
  case LMSOLVE_HERMITIAN_INDEFINITE_UPPER:
  case LMSOLVE_HERMITIAN_INDEFINITE_LOWER: {
    PetscInt     n = PetscMin(dots->k, dots->m);
    PetscBLASInt _n;
    PetscBLASInt _m;
    PetscBLASInt _lda;
    PetscBLASInt _info;
    PetscCall(PetscBLASIntCast(dots->m, &_m));
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(dots->lda, &_lda));
#if PetscDefined(USE_COMPLEX)
    if (!dots->work) {
      PetscScalar work = 0;
      PetscCallBLAS("LAPACKhetrf", LAPACKhetrf_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_m, dots->factorization, &_lda, dots->pivots, &work, &dots->lwork, &_info));
      PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XHETRF INFO=%d", (int)_info);
      dots->lwork = (PetscBLASInt)PetscRealPart(work);
      PetscCall(PetscMalloc1(dots->lwork, &dots->work));
    }
    PetscCallBLAS("LAPACKhetrf", LAPACKhetrf_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_n, dots->factorization, &_lda, dots->pivots, dots->work, &dots->lwork, &_info));
    PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XHETRF INFO=%d", (int)_info);
#else
    if (!dots->work) {
      PetscScalar work = 0;
      PetscCallBLAS("LAPACKsytrf", LAPACKsytrf_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_m, dots->factorization, &_lda, dots->pivots, &work, &dots->lwork, &_info));
      PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XSYTRF INFO=%d", (int)_info);
      dots->lwork = (PetscBLASInt)PetscRealPart(work);
      PetscCall(PetscMalloc1(dots->lwork, &dots->work));
    }
    PetscCallBLAS("LAPACKsytrf", LAPACKsytrf_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_n, dots->factorization, &_lda, dots->pivots, dots->work, &dots->lwork, &_info));
    PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XSYTRF INFO=%d", (int)_info);
#endif
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianSolve(LMGramian dots, PetscInt oldest, PetscInt next, LMSolveType solve_type, PetscScalar b[], PetscBool hermitian_transpose)
{
  PetscInt    dots_oldest = PetscMax(0, dots->k - dots->m);
  PetscInt    dots_next   = dots->k;
  PetscInt    offset      = oldest - dots_oldest;
  LMBlockType block_type  = LMBlockTypeFromSolveType(solve_type);

  PetscFunctionBegin;
  PetscCheck(oldest >= dots_oldest && next <= dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid indices");
  switch (solve_type) {
  case LMSOLVE_LU: {
    PetscCheck(oldest == dots_oldest && next == dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot solve on subset of indices");
    PetscCall(LMGramianComputeFactorization(dots, solve_type));
    PetscInt     n = PetscMin(dots->k, dots->m);
    PetscBLASInt _n;
    PetscBLASInt _one = 1;
    PetscBLASInt _lda;
    PetscBLASInt _info;
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(dots->lda, &_lda));
    PetscCallBLAS("LAPACKgetrs", LAPACKgetrs_(hermitian_transpose ? "C" : "N", &_n, &_one, dots->factorization, &_lda, dots->pivots, b, &_lda, &_info));
    PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XGETRS INFO=%d", (int)_info);
    break;
  }
  case LMSOLVE_UPPER_TRIANGLE:
  case LMSOLVE_LOWER_TRIANGLE: {
    PetscInt     lda = dots->lda;
    PetscScalar *A   = &dots->full[offset + offset * lda];
    PetscBLASInt lda_b;
    PetscBLASInt m_b;
    PetscBLASInt one_b = 1;

    b = &b[offset];
    PetscCall(PetscBLASIntCast(dots->lda, &lda_b));
    PetscCall(PetscBLASIntCast(next - oldest, &m_b));
    PetscCallBLAS("BLAStrsv_", BLAStrsv_(solve_type == LMSOLVE_UPPER_TRIANGLE ? "U" : "L", hermitian_transpose ? "C" : "N", "N", &m_b, A, &lda_b, b, &one_b));
    break;
  }
  case LMSOLVE_DIAGONAL: {
    if (hermitian_transpose) {
      for (PetscInt i = oldest - dots_oldest; i < next - dots_oldest; i++) b[i] /= PetscConj(dots->diagonal[i]);
    } else {
      for (PetscInt i = oldest - dots_oldest; i < next - dots_oldest; i++) b[i] /= dots->diagonal[i];
    }
    break;
  }
  case LMSOLVE_HERMITIAN_INDEFINITE_UPPER:
  case LMSOLVE_HERMITIAN_INDEFINITE_LOWER: {
    PetscCheck(oldest == dots_oldest && next == dots_next, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot solve on subset of indices");
    PetscCall(LMGramianComputeFactorization(dots, solve_type));
    PetscInt     n = PetscMin(dots->k, dots->m);
    PetscBLASInt _n;
    PetscBLASInt _one = 1;
    PetscBLASInt _lda;
    PetscBLASInt _info;
    PetscCall(PetscBLASIntCast(n, &_n));
    PetscCall(PetscBLASIntCast(dots->lda, &_lda));
#if PetscDefined(USE_COMPLEX)
    PetscCallBLAS("LAPACKhetrs", LAPACKhetrs_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_n, &_one, dots->factorization, &_lda, dots->pivots, b, &_lda, &_info));
#else
    PetscCallBLAS("LAPACKsytrs", LAPACKsytrs_(block_type == LMBLOCK_UPPER_TRIANGLE ? "U" : "L", &_n, &_one, dots->factorization, &_lda, dots->pivots, b, &_lda, &_info));
#endif
    PetscCheck(!_info, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in LAPACK routine XSYTRS INFO=%d", (int)_info);
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Not implemented");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianReset(LMGramian dots)
{
  PetscFunctionBegin;
  if (dots) {
    dots->k = 0;
    for (PetscInt i = 0; i < LMBLOCK_END; i++) dots->status[i].next = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianInsertDiagonalValue(LMGramian dots, PetscInt i, PetscScalar v)
{
  PetscFunctionBegin;
  PetscInt oldest = PetscMax(0, dots->k - dots->m);
  PetscInt next   = dots->k;
  PetscInt idx = i - oldest;
  PetscCheck(i >= oldest && i < next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting value %d out of range [%d, %d)", (int)i, (int)oldest, (int)next);
  PetscCall(LMGramianAllocateDiagonal(dots));
  dots->diagonal[idx]                 = v;
  if (i == dots->status[LMBLOCK_DIAGONAL].next) {
    dots->status[LMBLOCK_DIAGONAL].next++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode LMGramianGetDiagonalValue(LMGramian dots, PetscInt i, PetscScalar *v)
{
  PetscFunctionBegin;
  PetscInt oldest = PetscMax(0, dots->k - dots->m);
  PetscInt next   = dots->k;
  PetscInt idx = i - oldest;
  PetscCheck(i >= oldest && i < next, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting value %d out of range [%d, %d)", (int)i, (int)oldest, (int)next);
  PetscCheck(dots->diagonal, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Diagonal values not allocated yet");
  *v = dots->diagonal[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}
