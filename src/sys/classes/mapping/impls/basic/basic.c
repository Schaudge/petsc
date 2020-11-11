#include <petsc/private/imbasicimpl.h>

PetscErrorCode IMCreate_Basic(IM m)
{
  IM_Basic       *mb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(m, &mb);CHKERRQ(ierr);
  ierr = IMInitializeBasic_Private(mb);CHKERRQ(ierr);
  m->allowedMap = PETSC_TRUE;
  m->data = (void *)mb;
  m->ops->destroy = IMDestroy_Basic;
  m->ops->setup = IMSetup_Basic;
  m->ops->permute = IMPermute_Basic;
  m->ops->getindices = IMGetIndices_Basic;
  PetscFunctionReturn(0);
}

PetscErrorCode IMDestroy_Basic(IM m)
{
  IM_Basic       *mb = (IM_Basic *)m->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMDestroyBasic_Private(mb);CHKERRQ(ierr);
  ierr = PetscFree(mb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMSetup_Basic(IM m)
{
  PetscInt       i, min, max;
  IM_Basic       *mb = (IM_Basic *)m->data;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m->nIdx[IM_LOCAL]) {
    min = max = m->idx[0];
    for (i = 1; i < m->nIdx[IM_LOCAL]; ++i) {
      max = max < m->idx[i] ? m->idx[i] : max;
      min = min > m->idx[i] ? m->idx[i] : min;
    }
    mb->min[IM_LOCAL] = min;
    mb->max[IM_LOCAL] = max;
  }
  ierr = PetscObjectGetComm((PetscObject)m, &comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&(mb->min[IM_LOCAL]), &(mb->min[IM_GLOBAL]), 1, MPIU_INT, MPI_MIN, comm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&(mb->max[IM_LOCAL]), &(mb->max[IM_GLOBAL]), 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IMGetIndices_Basic(IM m, const PetscInt *idx[])
{
  PetscFunctionBegin;
  *idx = m->idx;
  PetscFunctionReturn(0);
}

/* Must take IMBASIC as input for both, and m must be IM_ARRAY since permuting interval makes no sense here */
PetscErrorCode IMPermute_Basic(IM m, IM pm)
{
  PetscInt       *newIdx;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(m->nIdx[IM_LOCAL], &newIdx);CHKERRQ(ierr);
  /* copy every value in case the permuation doesn't change all of them */
  ierr = PetscArraycpy(newIdx, m->idx, m->nIdx[IM_LOCAL]);CHKERRQ(ierr);
  for (i = 0; i < m->nIdx[IM_LOCAL]; ++i) newIdx[i] = m->idx[pm->idx[i]];
  ierr = PetscArraycpy(m->idx, newIdx, m->nIdx[IM_LOCAL]);CHKERRQ(ierr);
  ierr = PetscFree(newIdx);CHKERRQ(ierr);
  m->sorted[IM_LOCAL] = IM_UNKNOWN;
  PetscFunctionReturn(0);
}

PetscErrorCode IMBasicCreateFromIndices(MPI_Comm comm, PetscInt n, const PetscInt idx[], PetscCopyMode mode, IM *mb)
{
  IM             mt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = IMCreate(comm, &mt);CHKERRQ(ierr);
  ierr = IMSetType(mt, IMBASIC);CHKERRQ(ierr);
  ierr = IMSetIndices(mt, n, idx, mode);CHKERRQ(ierr);
  ierr = IMSetUp(mt);CHKERRQ(ierr);
  *mb = mt;
  PetscFunctionReturn(0);
}
