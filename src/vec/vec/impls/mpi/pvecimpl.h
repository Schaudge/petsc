
#if !defined(__PVECIMPL)
#define __PVECIMPL

#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/deviceimpl.h>

typedef struct {
  PetscInt insertmode;
  PetscInt count;
  PetscInt bcount;
} VecAssemblyHeader;

typedef struct {
  PetscInt    *ints;
  PetscInt    *intb;
  PetscScalar *scalars;
  PetscScalar *scalarb;
  char         pendings;
  char         pendingb;
} VecAssemblyFrame;

typedef struct {
  VECHEADER
  PetscInt   nghost;      /* number of ghost points on this process */
  Vec        localrep;    /* local representation of vector */
  VecScatter localupdate; /* scatter to update ghost values */

  PetscBool          assembly_subset;     /* Subsequent assemblies will set a subset (perhaps equal) of off-process entries set on first assembly */
  PetscBool          first_assembly_done; /* Is the first time assembly done? */
  PetscBool          use_status;          /* Use MPI_Status to determine number of items in each message */
  PetscMPIInt        nsendranks;
  PetscMPIInt        nrecvranks;
  PetscMPIInt       *sendranks;
  PetscMPIInt       *recvranks;
  VecAssemblyHeader *sendhdr, *recvhdr;
  VecAssemblyFrame  *sendptrs; /* pointers to the main messages */
  MPI_Request       *sendreqs;
  MPI_Request       *recvreqs;
  PetscSegBuffer     segrecvint;
  PetscSegBuffer     segrecvscalar;
  PetscSegBuffer     segrecvframe;
#if defined(PETSC_HAVE_NVSHMEM)
  PetscBool use_nvshmem; /* Try to use NVSHMEM in communication of, for example, VecNorm */
#endif

  /* COO fields, assuming m is the vector's local size */
  PetscCount  coo_n;
  PetscCount  tot1;  /* Total local entries in COO arrays */
  PetscCount *jmap1; /* [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats in COO arrays */
  PetscCount *perm1; /* [tot1]: permutation array for local entries */

  PetscCount  nnz2;  /* Unique entries in recvbuf */
  PetscCount *imap2; /* [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in the vector */
  PetscCount *jmap2; /* [nnz2+1] */
  PetscCount *perm2; /* [recvlen] */

  PetscSF      coo_sf;
  PetscCount   sendlen, recvlen;  /* Lengths (in unit of PetscScalar) of send/recvbuf */
  PetscCount  *Cperm;             /* [sendlen]: permutation array to fill sendbuf[]. 'C' for communication */
  PetscScalar *sendbuf, *recvbuf; /* Buffers for remote values in VecSetValuesCOO() */
} Vec_MPI;

PETSC_INTERN PetscErrorCode VecDot_MPI(Vec, Vec, PetscManagedScalar, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMDot_MPI(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecTDot_MPI(Vec, Vec, PetscManagedScalar, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMTDot_MPI(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecNorm_MPI(Vec, NormType, PetscManagedReal, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMax_MPI(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecMin_MPI(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDestroy_MPI(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecView_MPI_Binary(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Draw_LG(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_Socket(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_HDF5(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecView_MPI_ADIOS(Vec, PetscViewer);
PETSC_EXTERN PetscErrorCode VecView_MPI(Vec, PetscViewer);
PETSC_INTERN PetscErrorCode VecGetSize_MPI(Vec, PetscInt *);
PETSC_INTERN PetscErrorCode VecGetValues_MPI(Vec, PetscInt, const PetscInt[], PetscScalar[]);
PETSC_INTERN PetscErrorCode VecSetValues_MPI(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode VecSetValuesBlocked_MPI(Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode);
PETSC_INTERN PetscErrorCode VecAssemblyBegin_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyEnd_MPI(Vec);
PETSC_INTERN PetscErrorCode VecAssemblyReset_MPI(Vec);
PETSC_INTERN PetscErrorCode VecCreate_MPI_Private(Vec, PetscBool, PetscInt, const PetscScalar[], PetscDeviceContext);
PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecDuplicate_MPI(Vec, Vec *, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecResetArray_MPI(Vec, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecPlaceArray_MPI(Vec, const PetscScalar *, PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSetPreallocationCOO_MPI(Vec, PetscCount, const PetscInt[], PetscDeviceContext);
PETSC_INTERN PetscErrorCode VecSetValuesCOO_MPI(Vec, const PetscScalar[], InsertMode, PetscDeviceContext);

static inline PetscErrorCode VecMXDot_MPI_Standard(Vec xin, PetscManagedInt nv, const Vec y[], PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*VecMXDot_SeqFn)(Vec, PetscManagedInt, const Vec[], PetscManagedScalar, PetscDeviceContext)) {
  PetscInt *nvptr;

  PetscFunctionBegin;
  PetscCall(VecMXDot_SeqFn(xin, nv, y, z, dctx));
  PetscCall(PetscManagedIntGetArray(dctx, nv, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ, PETSC_TRUE, &nvptr));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx, z, nvptr, MPIU_SUM, (PetscObject)xin));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecXDot_MPI_Standard(Vec xin, Vec yin, PetscManagedScalar z, PetscDeviceContext dctx, PetscErrorCode (*VecXDot_SeqFn)(Vec, Vec, PetscManagedScalar, PetscDeviceContext)) {
  const PetscInt one = 1;

  PetscFunctionBegin;
  PetscCall(VecXDot_SeqFn(xin, yin, z, dctx));
  PetscCall(PetscDeviceContextAllReduceManagedScalar_Internal(dctx, z, &one, MPIU_SUM, (PetscObject)xin));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecMinMax_MPI_Standard(Vec xin, PetscManagedInt idx, PetscManagedReal z, PetscDeviceContext dctx, PetscErrorCode (*VecMinMax_SeqFn)(Vec, PetscManagedInt, PetscManagedReal, PetscDeviceContext), const MPI_Op ops[2]) {
  PetscFunctionBegin;
  /* Find the local max */
  PetscCall(VecMinMax_SeqFn(xin, idx, z, dctx));
  if (PetscDefined(HAVE_MPIUNI)) PetscFunctionReturn(0);
  /* Find the global max */
  if (idx) {
    PetscReal *zptr;
    PetscInt  *idxptr;

    PetscCall(PetscManagedRealGetArray(dctx, z, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &zptr));
    PetscCall(PetscManagedIntGetArray(dctx, idx, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &idxptr));
    {
      struct {
        PetscReal v;
        PetscInt  i;
      } in = {*zptr, *idxptr + xin->map->rstart};

      PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &in, 1, MPIU_REAL_INT, ops[0], PetscObjectComm((PetscObject)xin)));
      *zptr   = in.v;
      *idxptr = in.i;
    }
  } else {
    const PetscInt one = 1;

    /* User does not need idx */
    PetscCall(PetscDeviceContextAllReduceManagedReal_Internal(dctx, z, &one, ops[1], (PetscObject)xin));
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecDotNorm2_MPI_Standard(Vec s, Vec t, PetscManagedScalar dp, PetscManagedScalar nm, PetscDeviceContext dctx, PetscErrorCode (*VecDotNorm2_SeqFn)(Vec, Vec, PetscManagedScalar, PetscManagedScalar, PetscDeviceContext)) {
  PetscScalar *dpptr, *nmptr;

  PetscFunctionBegin;
  PetscCall(VecDotNorm2_SeqFn(s, t, dp, nm, dctx));
  PetscCall(PetscManagedScalarGetArray(dctx, dp, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &dpptr));
  PetscCall(PetscManagedScalarGetArray(dctx, nm, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &nmptr));
  {
    PetscScalar sum[] = {*dpptr, *nmptr};

    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &sum, 2, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)s)));
    *dpptr = sum[0];
    *nmptr = sum[1];
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecNorm_MPI_Standard(Vec xin, NormType type, PetscManagedReal z, PetscDeviceContext dctx, PetscErrorCode (*VecNorm_SeqFn)(Vec, NormType, PetscManagedReal, PetscDeviceContext)) {
  PetscReal *zptr;
  PetscInt   zn = 1;
  MPI_Op     op = MPIU_SUM;

  PetscFunctionBegin;
  PetscCall(VecNorm_SeqFn(xin, type, z, dctx));
  switch (type) {
  case NORM_1_AND_2:
    // the 2 norm needs to be squared below before being summed, but NORM_1_AND_2 stores the
    // 2-norm in the second slot, so increment zn here
    zn = 2;
  case NORM_2:
    PetscCall(PetscManagedRealGetArray(dctx, z, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &zptr));
    // use zn to index here since NORM_1_AND_2 will increment it
    zptr[zn - 1] *= zptr[zn - 1];
  case NORM_1:
  case NORM_FROBENIUS: break;
  case NORM_INFINITY: op = MPIU_MAX; break;
  }
  PetscCall(PetscDeviceContextAllReduceManagedReal_Internal(dctx, z, &zn, op, (PetscObject)xin));
  if (type == NORM_2 || type == NORM_FROBENIUS || type == NORM_1_AND_2) {
    PetscCall(PetscManagedRealGetArray(dctx, z, PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE, PETSC_TRUE, &zptr));
    zptr[type == NORM_1_AND_2] = PetscSqrtReal(zptr[type == NORM_1_AND_2]);
  }
  PetscFunctionReturn(0);
}
#endif
