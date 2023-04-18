
#include <petscblaslapack.h>
#include <petscdevicetypes.h>
#include <petsc/private/cupmblasinterface.hpp>

namespace Petsc
{

namespace device
{

namespace cupm
{

namespace impl
{

  template <device::cupm::DeviceType T>
  struct _BlasInterface : BlasInterface<T> {
    public:
      static PetscErrorCode CUPMgemm_(PetscDeviceContext dctx, const char *transa, const char *transb, const PetscBLASInt *m, const PetscBLASInt *n, const PetscBLASInt *k, const PetscScalar *alpha, const PetscScalar *A, const PetscBLASInt *lda, const PetscScalar *B, const PetscBLASInt *ldb, const PetscScalar *beta, PetscScalar *C, const PetscBLASInt *ldc)
      {
        PetscFunctionBegin;

        cupmBlasInt_t cm, cn, ck;
        PetscCall(PetscCUPMBlasIntCast(m, &cm));
        PetscCall(PetscCUPMBlasIntCast(n, &cn));
        PetscCall(PetscCUPMBlasIntCast(k, &ck));

        auto transpose_A = (transa[0] == 'T') || (transa[0] == 't');
        auto transpose_B = (transb[0] == 'T') || (transb[0] == 't');

        cupmBlasHandle_t   handle;
        PetscCall(GetHandlesFrom_(dctx, &handle, NULL, NULL));
        PetscCallCUPMBLAS(cupmBlasXgemm(handle, transpose_A ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, transpose_B ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, m, n, k, alpha, A, (PetscInt) *lda, B, (PetscInt) *ldb, beta, C, (PetscInt) *ldc));

        PetscFunctionReturn(PETSC_SUCCESS);
      }

    private:
      PETSC_CUPMBLAS_IMPL_CLASS_HEADER(T);

      PETSC_NODISCARD static constexpr const char *cupmBlasName() noexcept { return T == DeviceType::CUDA ? "cuBLAS" : "hipBLAS"; }
  };



} // namespace impl

} // namespace cupm

} // namespace device

} // namespace Petsc

PetscErrorCode PetscMemtypeGEMM(PetscMemType memtype, const char *transa, const char *transb, PetscInt m, const PetscInt n, const PetscInt k, PetscScalar alpha, const PetscScalar *A, PetscInt lda, const PetscScalar *B, PetscInt ldb, PetscScalar beta, PetscScalar *C, PetscInt ldc)
{
  PetscFunctionBegin;
  switch (memtype) {
  case PETSC_MEMTYPE_HOST:
    {
      PetscBLASInt lda_blas, ldb_blas, ldc_blas, m_blas, n_blas, k_blas;
      PetscCall(PetscBLASIntCast(m, &m_blas));
      PetscCall(PetscBLASIntCast(n, &n_blas));
      PetscCall(PetscBLASIntCast(k, &k_blas));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCall(PetscBLASIntCast(ldb, &ldb_blas));
      PetscCall(PetscBLASIntCast(ldc, &ldc_blas));
      PetscCallBLAS("BLASgemm", BLASgemm_(transa, transb, &m_blas, &n_blas, &k_blas, &alpha, A, &lda_blas, B, &ldb_blas, &beta, C, &ldc_blas));
    }
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported");
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscMemtypeTRSM(PetscMemType memtype, const char *right_or_left, const char *upper_or_lower, const char *transpose_type, const char *unit_or_not, PetscInt m, PetscInt n, PetscScalar alpha, const PetscScalar *A, PetscInt lda, PetscScalar *B, PetscInt ldb)
{
  PetscFunctionBegin;
  switch (memtype) {
  case PETSC_MEMTYPE_HOST:
    {
      PetscBLASInt lda_blas, ldb_blas, m_blas, n_blas;
      PetscCall(PetscBLASIntCast(m, &m_blas));
      PetscCall(PetscBLASIntCast(n, &n_blas));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCall(PetscBLASIntCast(ldb, &ldb_blas));
      PetscCallBLAS("BLAStrsm", BLAStrsm_(right_or_left, upper_or_lower, transpose_type, unit_or_not, &m_blas, &n_blas, &alpha, A, &lda_blas, B, &ldb_blas));
    }
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported");
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
