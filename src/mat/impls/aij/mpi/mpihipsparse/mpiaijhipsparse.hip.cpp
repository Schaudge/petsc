#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpihipsparse/mpihipsparsematimpl.h>

PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b               = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJHIPSPARSE * hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)b->spptr;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJHIPSPARSE matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatPinToCPU(b->A,B->pinnedtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatPinToCPU(b->B,B->pinnedtocpu);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetFormat(b->A,MAT_HIPSPARSE_MULT,hipsparseStruct->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetFormat(b->B,MAT_HIPSPARSE_MULT,hipsparseStruct->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->A,hipsparseStruct->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetHandle(b->B,hipsparseStruct->handle);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->A,hipsparseStruct->stream);CHKERRQ(ierr);
  ierr = MatHIPSPARSESetStream(b->B,hipsparseStruct->stream);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  /*
     This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.
  */
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  /*
     This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.
  */
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,xx);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJHIPSPARSE(Mat A,Vec xx,Vec yy)
{
  /* This multiplication sequence is different sequence
     than the CPU version. In particular, the diagonal block
     multiplication kernel is launched in one stream. Then,
     in a separate stream, the data transfers from DeviceToHost
     (with MPI messaging in between), then HostToDevice are
     launched. Once the data transfer stream is synchronized,
     to ensure messaging is complete, the MatMultAdd kernel
     is launched in the original (MatMult) stream to protect
     against race conditions.

     This sequence should only be called for GPU computation. */
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
  ierr = VecScatterInitializeForGPU(a->Mvctx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterFinalizeForGPU(a->Mvctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatHIPSPARSESetFormat_MPIAIJHIPSPARSE(Mat A,MatHIPSPARSEFormatOperation op,MatHIPSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a               = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE * hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT_DIAG:
    hipsparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_MULT_OFFDIAG:
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparseStruct->diagGPUMatFormat    = format;
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatHIPSPARSEFormatOperation. Only MAT_HIPSPARSE_MULT_DIAG, MAT_HIPSPARSE_MULT_DIAG, and MAT_HIPSPARSE_MULT_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPIAIJHIPSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  MatHIPSPARSEStorageFormat format;
  PetscErrorCode           ierr;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE       *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)a->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPIAIJHIPSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_hipsparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT_DIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_hipsparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_MULT_OFFDIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_hipsparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV",
                            "MatHIPSPARSESetFormat",MatHIPSPARSEStorageFormats,(PetscEnum)hipsparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatHIPSPARSESetFormat(A,MAT_HIPSPARSE_ALL,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJHIPSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *mpiaij;

  PetscFunctionBegin;
  mpiaij = (Mat_MPIAIJ*)A->data;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQHIP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJHIPSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a              = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJHIPSPARSE *hipsparseStruct = (Mat_MPIAIJHIPSPARSE*)a->spptr;
  hipError_t        err;
  hipsparseStatus_t   stat;

  PetscFunctionBegin;
  try {
    ierr = MatHIPSPARSEClearHandle(a->A);CHKERRQ(ierr);
    ierr = MatHIPSPARSEClearHandle(a->B);CHKERRQ(ierr);
    stat = hipsparseDestroy(hipsparseStruct->handle);CHKERRHIPSPARSE(stat);
    if (hipsparseStruct->stream) {
      err = hipStreamDestroy(hipsparseStruct->stream);CHKERRHIP(err);
    }
    delete hipsparseStruct;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJHIPSPARSE error: %s", ex);
  }
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJHIPSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  Mat_MPIAIJHIPSPARSE * hipsparseStruct;
  hipsparseStatus_t   stat;

  PetscFunctionBegin;
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECHIP,&A->defaultvectype);CHKERRQ(ierr);

  a        = (Mat_MPIAIJ*)A->data;
  a->spptr = new Mat_MPIAIJHIPSPARSE;

  hipsparseStruct                      = (Mat_MPIAIJHIPSPARSE*)a->spptr;
  hipsparseStruct->diagGPUMatFormat    = MAT_HIPSPARSE_CSR;
  hipsparseStruct->offdiagGPUMatFormat = MAT_HIPSPARSE_CSR;
  hipsparseStruct->stream              = 0;
  stat = hipsparseCreate(&(hipsparseStruct->handle));CHKERRHIPSPARSE(stat);

  A->ops->assemblyend    = MatAssemblyEnd_MPIAIJHIPSPARSE;
  A->ops->mult           = MatMult_MPIAIJHIPSPARSE;
  A->ops->multadd        = MatMultAdd_MPIAIJHIPSPARSE;
  A->ops->multtranspose  = MatMultTranspose_MPIAIJHIPSPARSE;
  A->ops->setfromoptions = MatSetFromOptions_MPIAIJHIPSPARSE;
  A->ops->destroy        = MatDestroy_MPIAIJHIPSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJHIPSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatHIPSPARSESetFormat_C",  MatHIPSPARSESetFormat_MPIAIJHIPSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatCreateAIJHIPSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to GPUs and use the HIPSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJHIPSPARSE, MATAIJHIPSPARSE
@*/
PetscErrorCode  MatCreateAIJHIPSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJHIPSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJHIPSPARSE - MATMPIAIJHIPSPARSE = "aijhipsparse" = "mpiaijhipsparse" - A matrix type to be used for sparse matrices.

   A matrix type type whose data resides on GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. 
   All matrix calculations are performed using the HIPSPARSE library.

   This matrix type is identical to MATSEQAIJHIPSPARSE when constructed with a single process communicator,
   and MATMPIAIJHIPSPARSE otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijhipsparse - sets the matrix type to "mpiaijhipsparse" during a call to MatSetFromOptions()
.  -mat_hipsparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
.  -mat_hipsparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

 .seealso: MatCreateAIJHIPSPARSE(), MATSEQAIJHIPSPARSE, MatCreateSeqAIJHIPSPARSE(), MatHIPSPARSESetFormat(), MatHIPSPARSEStorageFormat, MatHIPSPARSEFormatOperation
M
M*/
